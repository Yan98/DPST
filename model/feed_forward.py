import torch.nn as nn
import torch

class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation = nn.SiLU()):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1, distance = False, gate = False, concat = False):
        super(PositionwiseFeedForward, self).__init__()
        
        d_in = d_model * (1 + distance + concat) if distance else d_model
        if distance or gate:
            self.w_1 = GLU(d_in,d_ff, nn.SiLU()) 
        else:
            self.w_1 = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(True)
                )
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.w_2(self.w_1(x))) 
        return  x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, residual = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.residual = residual
    def forward(self, x, **kwargs):
        y = self.fn(self.norm(x), **kwargs)
        if self.residual:
            y = y + x
        return y