import torch.nn as nn
from .feed_forward import PositionwiseFeedForward,PreNorm
from .embedding import RelativePositionBias,PositionalEmbedding
from pytorch3d import ops
import torch
import torch.nn.functional as F

class MultiHeadedSelfAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, ape = True):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.ape = ape
    
        self.to_qkv = nn.Linear(d_model, d_model * 3,bias = False)
        self.output_linear = nn.Linear(d_model, d_model,bias = False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, pos_fn, mask=None):
        
        batch_size = query.size(0)
        
        
        if self.ape:
            query = pos_fn(query)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k            
        query,key,value = self.to_qkv(query).view(batch_size, -1, self.h, self.d_k * 3).transpose(1, 2).chunk(3,-1)

        # 2) Apply attention on all the projected vectors in batch.
        scale =  query.size(-1) ** -0.5
        
        scores = torch.matmul(query, key.transpose(-2, -1))
        if not self.ape:
            scores = pos_fn(torch.matmul(query, key.transpose(-2, -1))) 
        scores = scores * scale
                
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        x =  torch.matmul(attn, value)        
        
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = self.dropout(self.output_linear(x))
        
        return x
    
class MultiHeadedCrossAttention(nn.Module):
    def __init__(self,h, d_model,d_hidden, local = True,dropout = 0.1, distance = False):
        super().__init__()

        self.h = h 
        self.d_k = d_hidden // h
        self.local = local
        self.to_q = nn.Linear(d_model,  d_hidden, bias=False)
        self.to_kv = nn.Linear(d_model, d_hidden * 2, bias = False)
        
       	self.pos = nn.Sequential(
            nn.Linear(d_hidden if distance == False else d_model,d_hidden),
            nn.ReLU(True),
            nn.Linear(d_hidden,d_hidden * 2),
            )
        
        self.output_linear = nn.Linear(d_hidden + (d_model if distance else 0), d_model,bias = False)
       	self.dropout = nn.Dropout(p=dropout)
        
    def forward(self,q,k,pos, mask = None, local_idx = None, local_len = None,local_shape = None, distance = None):
        scale = self.d_k ** -0.5
        
        q = self.to_q(q) #b l h
        k,v = self.to_kv(k).chunk(2,-1) #b l h
        b,i,j,_  = pos.size()
        pos_k, pos_v = self.pos(pos).view(b,i,j,self.h,self.d_k * 2).chunk(2,-1)
        
        q = q.view(b,i,self.h,self.d_k)
    
        if not self.local:
            k = k.view(b,j,self.h,self.d_k)[:,None,:,:,:]
            v = v.view(b,j,self.h,self.d_k)[:,None,:,:,:]
        else:
            k = ops.knn_gather(k, local_idx, local_len).view(*local_shape).view(b,i,j,self.h,self.d_k)
            v = ops.knn_gather(v, local_idx, local_len).view(*local_shape).view(b,i,j,self.h,self.d_k)
            
        k = k + pos_k
        v = v + pos_v    
        attn = torch.einsum("b i h d, b i j h d -> b i j h",q,k) * scale
        #attn = ((q[:,:, None,:] * k).view(b,i,j,self.h,self.d_k).sum(-1) + pos) * scale #b l k 2

        if mask != None:
            attn = attn.masked_fill_(mask,-1e9)
            
        attn = torch.nn.functional.softmax(attn,2)
        attn = self.dropout(attn)
        agg = torch.einsum("b i j h, b i j h d -> b i h d", attn, v).contiguous().view(b,i,-1)
        if distance != None:
            agg = torch.cat((agg,distance),-1)
        agg = self.dropout(self.output_linear(agg))
        return agg 
    
class DecoderLayer(nn.Module):
    
    def __init__(self,hidden, attn_heads,feed_forward_hidden,dropout, local_attn = True, global_attn = True, ape = True, down = 2):
        super().__init__()
        
        self.local_attn = local_attn
        self.global_attn = global_attn
        self.slf_attn = PreNorm(hidden, MultiHeadedSelfAttention(h=attn_heads, d_model=hidden, ape = ape))
        if local_attn:
            self.local_attention  = PreNorm(hidden,MultiHeadedCrossAttention(h=attn_heads, d_model=hidden, d_hidden = hidden//down, local = True, distance = True), residual= False if global_attn else True)
        if global_attn:
            self.global_attention = PreNorm(hidden, MultiHeadedCrossAttention(h=attn_heads, d_model=hidden, d_hidden = hidden//down, local = False))
        self.feed_forward = PreNorm(hidden, PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, distance = False, gate = False, concat= False))
        self.pos_emb = PositionalEmbedding(hidden, 50) if ape else RelativePositionBias(causal = True, num_buckets = 12, max_distance = 48, heads = attn_heads) 
        
    def forward(self,dec_input, enc_output, distance, slf_attn_mask, global_dist,local_dist,local_idx, local_len,local_shape,p_len_mask):
        dec_input = self.slf_attn(dec_input,pos_fn = self.pos_emb, mask = slf_attn_mask)
        if self.local_attn:
            dec_input_local = self.local_attention(dec_input, k = enc_output, pos = local_dist, local_idx = local_idx, local_len = local_len,local_shape = local_shape, distance = distance)       
        if self.global_attn:
            dec_input = self.global_attention(dec_input, k = enc_output,pos = global_dist, mask = p_len_mask)
        if self.local_attn and self.global_attn:
            dec_input = dec_input + dec_input_local
        elif self.local_attn:
            dec_input = dec_input_local
        dec_input = self.feed_forward(dec_input)
        return dec_input
                        

         
         
         
         
         
         
         
        
      
        
        
