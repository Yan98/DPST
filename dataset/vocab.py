import copy
from . import configuration

class ProteinVocab(object):
    def __init__(self):
        self.pad       = "<pad>"   #padding
        self.pad_index = 0
        
        self.sos       = "<sos>"   #start
        self.sos_index = 1
        
        self.eos       = "<eos>"   #end
        self.eos_index = 2
        
        self.code          ={x:y for y,x in enumerate(configuration.vocab_reserve,3)}
        self.reverse_code  ={y:x for x,y in self.code.items()}
        self.code_all      = copy.deepcopy(self.code)
        self.code_all.update({
            self.pad : self.pad_index,
            self.sos : self.sos_index,
            self.eos : self.eos_index
            })
        
        self.aa2mass_num = {self.code_all[i] : j for i,j in configuration.aa2mass.items()}
        
    def encode(self,peptide):
        aa = [self.code[i] for i in peptide]
        aa = [self.sos_index] + aa + [self.eos_index]
        return aa

    def decode(self,indexs):
        return [self.reverse_code[i] for i in indexs]

    def pad_seq(self,aa,target_length):
        if len(aa) > target_length:
            return aa[:target_length]
        aa = aa + [self.pad_index] * (target_length - len(aa))
        return aa
    
    def size(self):
        return len(self.code_all)
        
        
    
