#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from  einops import rearrange
import numpy as np 
from  pytorch3d import ops
from .feed_forward import PositionwiseFeedForward,PreNorm  


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        
        norm = self.kwargs["norm"]
        
        normalized = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        org  = torch.cat([fn(inputs * norm) for fn in self.embed_fns], -1)
        
        return torch.cat((normalized,org[...,1:]),-1)


def get_embedder(multires = 10, i=0):
    if i == -1:
        return nn.Identity(), 1
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 1,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
                "norm" : 3000,
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
  
embeder = get_embedder()[0]

            
class Encoder(nn.Module):
    def __init__(self, hidden,attn_heads,feed_forward_hidden, n_layers,dropout = 0.1,kq = 8, kv = 12, value_attn = True, save_memory = False):
        super().__init__()
        self.hidden = hidden
        self.kq      = kq
        self.kv      = kv
        self.norm   = 3000
        self.value_attn = value_attn
        self.pret   = nn.Sequential(
            nn.Linear(21 + 20 + 1 + 21 + 20,feed_forward_hidden,bias = False),
            nn.ReLU(True),
            nn.Linear(feed_forward_hidden,hidden),
            nn.ReLU(True),
            )
        self.layers = nn.ModuleList([EncoderLayer(hidden, attn_heads,feed_forward_hidden,dropout, value_attn = value_attn, save_memory = save_memory) for _ in range(n_layers)])
        self.n_layers = n_layers
        self.final_block = PreNorm(hidden,PositionwiseFeedForward(hidden, feed_forward_hidden))      
        
       	self.pos_mlp = nn.Sequential(
            nn.Linear(20 + 1 + 25 + 20, hidden),
            nn.ReLU(True),
            PositionwiseFeedForward(hidden,feed_forward_hidden)
            )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self,data):
        p_len = data["p_len"]
        num_of_point = p_len.max()
        mz   =  data["mz"] / self.norm
        intensity = data["intensity"]
        total = data["h_mass"].sum(-1)[:,[0]].expand(-1,num_of_point) / self.norm
        
        with torch.no_grad():
            pos_encoding = embeder(mz[:,:,None])
            total_encoding = embeder(total[:,:,None])
            x = torch.cat((pos_encoding, intensity[:,:,None],total_encoding),-1)
            
            knn_dist, knn_idx, _ = ops.knn_points(mz.unsqueeze(-1), mz.unsqueeze(-1),p_len,p_len, K = self.kq, return_nn = False)
            knn_dist = knn_dist.sqrt() * torch.sign(
                (ops.knn_gather(mz.unsqueeze(-1), knn_idx,p_len) - mz.unsqueeze(-1).unsqueeze(-1)).squeeze(-1)
                )
            knn_dist = torch.cat((knn_dist.unsqueeze(-1),
                                  torch.FloatTensor([0] * 24 + [1]).to(p_len.device).view(1,1,1,25).expand(x.size(0),x.size(1),self.kq,-1))
                                 ,-1)
            aa_inf = get(mz,p_len,self.kv)
            if self.value_attn:
                knn_dist = torch.cat((knn_dist,aa_inf[0]),2) #b l k c
            pos = knn_dist[...,[0]]
            knn_dist = knn_dist[...,:-1]
            pos = embeder(pos) 
            knn_dist = torch.cat((pos,knn_dist),-1)
        
        knn_dist = self.dropout(self.pos_mlp(knn_dist).masked_fill(aa_inf[2],0))
        x = self.dropout(self.pret(x))        
        for n_layer in range(self.n_layers):
            x = self.layers[n_layer](x,knn_idx,knn_dist,p_len,aa_inf,self.kq if self.value_attn else None)
        x = self.dropout(self.final_block(x) + x)
        return x,aa_inf[2]        


class EncoderLayer(nn.Module):
    def __init__(self,hidden, attn_heads,feed_forward_hidden,dropout,value_attn,save_memory):
        super().__init__()
        self.self_att = PreNorm(hidden, EncoderAttention(h = attn_heads, d_model = hidden, pos_expand = 2, value_attn = value_attn, save_memory = save_memory))
        self.feed_forward = PreNorm(hidden, PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, gate = value_attn))
    def forward(self,x,idx,dist,p_len,aa_inf,sep):
        x = self.self_att(x,idx = idx, dist = dist,p_len = p_len, aa_inf = aa_inf, sep = sep)
        x = self.feed_forward(x)
        return x
        

class EncoderAttention(nn.Module):
    def __init__(self,h, d_model, pos_expand, value_attn, save_memory):
        super().__init__()
        
        self.h = h
        self.d_k = d_model // h
        self.value_attn = value_attn
        self.save_memory = save_memory
        
        if value_attn:
            self.to_qkvg = nn.Linear(d_model, d_model * 4, bias=False)
        else:
            self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        
        #position encoding 
        self.pos_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * pos_expand),
            nn.ReLU(True),
            nn.Linear(d_model * pos_expand, d_model * 2)
        )
        
        self.out_linear = nn.Linear(d_model,d_model,bias = False)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self,q,idx,dist,p_len,aa_inf, sep = 4):
        
        scale = self.d_k ** -0.5
        
        if self.value_attn:
            qk,vg = self.to_qkvg(q).chunk(2,2)
            q, k = qk.chunk(2,2)
        else:
            q,k,v = self.to_qkv(q).chunk(3, 2)
            
        k   = ops.knn_gather(k, idx,p_len)
        
        if self.value_attn:
            dist_k, dist_v = self.pos_mlp(dist).chunk(2,3)
            pos_q_k, pos_v_k = dist_k[:,:,:sep], dist_k[:,:,sep:]
            pos_q_v, pos_v_v = dist_v[:,:,:sep], dist_v[:,:,sep:]
            v,g  = gather(vg,aa_inf[1],p_len,aa_inf[2]).chunk(2,3)
            v    = ((g + pos_v_k).sigmoid() * (v + pos_v_v)).max(2)[0]
        else:
            pos_q_k, pos_q_v = self.pos_mlp(dist).chunk(2,3)
            
        v = ops.knn_gather(v, idx,p_len)
        
        b,i,j, _  = v.size()
        
        if self.save_memory:
            q  = q.view(b,i,self.h,self.d_k)    
            k  = (k + pos_q_k).view(b,i,j,self.h,self.d_k)
            attn = torch.einsum("b i h d, b i j h d -> b i j h", q, k) * scale
        else:
            attn = (q[:,:, None,:] * (k + pos_q_k)).view(b,i,j,self.h,self.d_k).sum(-1) * scale #b l k 2
        attn = torch.nn.functional.softmax(attn,2)
        attn = self.dropout(attn)
        
       	v = (v + pos_q_v).view(b,i,j,self.h,self.d_k) #b l k 2 d
        agg = torch.einsum('b i j h, b i j h d -> b i h d', attn, v).view(b,i,-1)
        agg = self.dropout(self.out_linear(agg))
        
        return agg
        
ions = [
 71.03711,
 156.10111,
 114.04293,
 115.02694,
 160.03065,
 129.04259,
 128.05858,
 57.02146,
 137.05891,
 113.08406,
 113.08406,
 128.09496,
 131.04049,
 147.06841,
 97.05276,
 87.03203,
 101.04768,
 186.07931,
 163.06333,
 99.06841,        
 ]

number_of_aa = len(ions)

ions  = ions + [-i for i in ions] + [1/2 * i for i in ions] + [-1/2 * i for i in ions]
extras = []

for i in range(4):
    extra = np.array([[0] * 25] * number_of_aa)
    for j in range(number_of_aa):
        extra[j,j] = 1
        extra[j,number_of_aa + i] = 1
        extra[j,24] = 1
    extras.append(extra)     
  
def get(mz,p_len, kv, norm = 3000,k = 3):
    b, l = mz.size()[:2]
    mz   = mz.view(b,l,1,1)
    ion  = torch.FloatTensor(ions).to(mz.device).view(1,1,80,1)/norm
    ion  = ion + mz 
    mask = torch.logical_or(ion > 1, ion < 0) #b l 80 1
    dist, idx, _ = ops.knn_points(ion.view(b, l * 80,1), mz.view(b,l,1), lengths2=p_len, K=k, return_nn = False)
    
    dist_sign = torch.sign(
        (ops.knn_gather(mz.view(b, l,1), idx, p_len).view(b,l,80,k) - ion.view(b,l,80,1)).squeeze(-1)
        ).view(b,l,-1,1)
    
    dist     = (- dist.sqrt()).masked_fill_(mask.view(b, l * 80,1), -10)
    
    dist_idx = torch.topk(dist.view(b,l,-1,1),kv,2)[1]
    e    = torch.FloatTensor(extras).to(mz.device).view(1, 1, 80,1,25).expand(b,l,-1,k,-1)
    dist = torch.cat((dist.view(b,l,-1,1).abs() * dist_sign,e.reshape(b,l,-1,25)),-1).view(b,l, -1,1 + 25)
    dist  = torch.gather(dist,2,dist_idx.expand(-1,-1,-1, 25 + 1 ))#b l kv 1
    mask  = torch.arange(l).to(mz.device).view(1,l,1,1).expand(b,-1,-1,-1) >= p_len.view(b,1,1,1)
    dist  = dist.masked_fill_(mask,0)
    idx = torch.gather(idx.view(b,l,-1,1),2,dist_idx).view(b,l,-1)
    
    return dist,idx,mask

def gather(x,idx,p_len,mask):
    x = ops.knn_gather(x,idx,p_len)
    x = x.masked_fill_(mask,0)
    return x 
  