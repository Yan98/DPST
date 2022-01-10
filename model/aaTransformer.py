import torch.nn as nn
import torch
import numpy as np
from .decoder import DecoderLayer
from .embedding import Embedding
from .encoder import Encoder, embeder
from pytorch3d import ops
from .feed_forward import PositionwiseFeedForward
    

def get_subsequent_mask(seq, sz_b, len_s):
    ''' For masking out the subsequent info. '''
    dia = 1
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, 1, len_s, len_s), device=seq.device), diagonal=dia)).bool() #1
    #if dia == 0:
    #    subsequent_mask[:,0,0] = True
    return subsequent_mask

class aaTransformer(nn.Module):
    def __init__(self, vocab_size, hidden=768, factor = 4,n_layers=12, attn_heads=12, dropout=0.1, local_attn = True, global_attn = True, value_attn = True, save_memory = False, ape = True, kq = 8, kv = 12, down = 2, first = 6):
        super().__init__()
        self.vocab_size    = vocab_size
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.vocab_size    = vocab_size
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.local_attn = local_attn
        self.global_attn = global_attn

        self.encoder       = Encoder(hidden,attn_heads, hidden * factor, n_layers, value_attn = value_attn, save_memory = save_memory, kq = kq, kv = kv)        

        self.embedding     = Embedding(vocab_size=vocab_size, embed_size=hidden)
       	
        if local_attn:
            self.locs = nn.Sequential(
                nn.Linear(21 + 20, hidden),
                nn.ReLU(True),
                PositionwiseFeedForward(hidden,hidden * factor),
                )
            
            self.distance  = nn.Sequential(
                nn.Linear(22, hidden),
                nn.ReLU(True),
                PositionwiseFeedForward(hidden,hidden * factor),
                )
        
        if global_attn:
            self.glocs = nn.Sequential(
                nn.Linear((20 + 1 + 20) * 2,hidden //2),
                nn.ReLU(True),
                PositionwiseFeedForward(hidden //2, hidden //2 * factor),
                )
        
        self.decoder = nn.ModuleList([DecoderLayer(hidden,attn_heads, hidden * factor, dropout, local_attn = local_attn and i < first, global_attn = global_attn, ape = ape, down = down) for i in range(n_layers)])
        
    def forward(self,data, norm = 3000):
                             #b,l,4,26  
        
        aa    = data["aa_input"]
        hmass = data["h_mass"]/norm
        mz    = data["mz"].unsqueeze(-1)/norm 
        
        ion_location = data["c_ion"]/norm  #b l 12
        
        with torch.no_grad():
            att_aa       = (aa > 0).unsqueeze(1).expand(-1, aa.size(1), -1).unsqueeze(1)
            att_aa       = torch.logical_and(att_aa, get_subsequent_mask(aa, aa.size(0),aa.size(1)))      
            
            batch, max_len, num_of_ion  = ion_location.size()
            
            if self.local_attn:
                distance = self.distance(data["distance"])        #b l 25
                ion_location = ion_location.reshape(-1,max_len * num_of_ion,1)
                knn_dist,knn_idx, _ = ops.knn_points(ion_location, mz, K = 2,lengths2=data["p_len"], return_nn = False)
                knn_dist = (- knn_dist.sqrt()).view(batch,max_len,num_of_ion,2,1)
                knn_dist  = knn_dist.masked_fill_(torch.logical_not(data["aa_mask"]).view(batch,max_len,1,1,1),0)
                knn_dist     = knn_dist.view(batch,max_len, num_of_ion * 2, -1)
                knn_dist  = embeder(knn_dist).view(batch,max_len,24,-1)
            else:
                distance = None
                knn_dist = None
                knn_idx  = None
                
            if self.global_attn:
                glocs = (hmass.unsqueeze(2) - mz.unsqueeze(1)).unsqueeze(4)
                glocs = embeder(glocs).view(batch,max_len,-1, (20 + 1 + 20) * 2)
        
        glocs = self.glocs(glocs) if self.global_attn else None
        knn_dist = self.locs(knn_dist) if self.local_attn else None    
        
        aa = self.embedding(aa)
        mass, p_len_mask  = self.encoder(data)
        p_len_mask = p_len_mask.view(batch,1,-1,1)
        
        for nth_layer in range(self.n_layers):
            aa = self.decoder[nth_layer](dec_input = aa, enc_output = mass, distance = distance, slf_attn_mask = att_aa, \
                                         global_dist = glocs,local_dist = knn_dist,local_idx = knn_idx, local_len = data["p_len"], \
                                             local_shape = (batch,max_len, num_of_ion * 2,-1),p_len_mask = p_len_mask)
        return aa
        
        
        