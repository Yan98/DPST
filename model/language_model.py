import torch.nn as nn
from .aaTransformer import aaTransformer
    
class Mass2Protein(nn.Module):
    def __init__(self,model: aaTransformer):
        super().__init__()
        self.model    = model
        self.mprotein =  nn.Linear(model.hidden,model.vocab_size)
        
    def forward(self, data):
        protein = self.model(data)
        mprotein   = self.mprotein(protein)
        return mprotein 