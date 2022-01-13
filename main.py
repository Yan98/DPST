#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataset import DenovoDataset, SMSDataset,collate_func_denovo
from trainer import DenovoTrainer,TrainerModel
from model import aaTransformer,Mass2Protein
from torch.utils.data import DataLoader
import torch
import os
import pytorch_lightning as pl 
from functools import partial
from argparse import ArgumentParser
import collections

dataname     =[
    ("cross.9high_80k.exclude_bacillus/",     "high.bacillus.PXD004565/"), #Bacillus
    ("cross.9high_80k.exclude_clambacteria/", "high.clambacteria.PXD004536/"), #c.endoloripes
    ("cross.9high_80k.exclude_honeybee/",     "high.honeybee.PXD004467/"), #Apis mellifera
    ("cross.9high_80k.exclude_human/",        "high.human.PXD004424/"),  #H. sapiens
    ("cross.9high_80k.exclude_mmazei/",       "high.mmazei.PXD004325/"), #Methanosarcina mazei
    ("cross.9high_80k.exclude_mouse/",        "high.mouse.PXD004948/"), # Mus musculus
    ("cross.9high_80k.exclude_ricebean/",     "high.ricebean.PXD005025/"),
    ("cross.9high_80k.exclude_tomato/",       "high.tomato.PXD004947/"), #Solanum lycopersicum
    ("cross.9high_80k.exclude_yeast/",        "high.yeast.PXD003868/"), #Saccharomyces cerevisiae
]


def main(args):

    cwd = os.getcwd()
    
    def write(director,name,*string):
        string = [str(i) for i in string]
        string = " ".join(string)
        with open(os.path.join(director,name),"a") as f:
            f.write(string + "\n")
    if args.output == None:
        store_dir = "output_" + str(args.fold)
        print = partial(write,cwd,"log_" + str(args.fold))
    else:
        store_dir = "output_" + str(args.output)
        print = partial(write,cwd,"log_" + str(args.output))
        
    os.makedirs(store_dir, exist_ok= True)
    
    print(args)
    
    for train,test in  [dataname[args.fold]]:
        
        datasets = []
        for name in os.listdir(os.path.join(PREFIX,train)):
            if not name.endswith(".repeat"):
                continue
            datasets = DenovoDataset(os.path.join(PREFIX,train,name), logfun = print)
            if "train" in name:
                train_loader  = DataLoader(datasets,batch_size=args.batch, shuffle= True, num_workers  = args.workers, pin_memory= True, collate_fn = collate_func_denovo, prefetch_factor = 2)                
            elif "valid" in name: #elif "test" in name:
                test_loader  = DataLoader(datasets,batch_size=args.batch, shuffle= True, num_workers  = args.workers, pin_memory= True, collate_fn = collate_func_denovo, prefetch_factor = 2)
        
        #For testing on diffirent organization
        #datasets      = DenovoDataset(os.path.join(PREFIX,test,"peaks.db.mgf"),transform = None, logfun = print)
        #test_loader   = DataLoader(datasets,batch_size=args.batch, shuffle= False, num_workers = args.workers, pin_memory= True, collate_fn = collate_func_denovo, prefetch_factor = 2)
        #128
        model         = aaTransformer(test_loader.dataset.vocab.size(),
                           hidden   =  args.hidden,
                           attn_heads = args.heads,
                           factor   = args.factor, 
                           n_layers = args.layers,
                           local_attn = args.local_attn,
                           global_attn = args.global_attn,
                           value_attn = args.value_attn,
                           save_memory = args.save_memory,
                           ape = args.ape,
                           kq = args.kq,
                           kv = args.kv,
                           down = args.down,
                           first = args.first
                           )
        model   = Mass2Protein(model)
        
        if args.checkpoints != None:
            model.load_state_dict(torch.load(args.checkpoints,map_location = torch.device("cpu")))
            
        CONFIG = collections.namedtuple('CONFIG', ['lr', 'logfun', 'pad_index', 'verbose_step', 'weight_decay', 'store_dir'])
        config = CONFIG(args.lr, print, datasets.vocab.pad_index, args.verbose_step, args.weight_decay,store_dir)
                
        model = TrainerModel(config, model)
        plt = pl.Trainer(max_epochs = args.epoch,num_nodes=args.num_nodes, gpus=args.gpus,accelerator= args.acce, val_check_interval = args.val_interval,  profiler= args.profiler)
        plt.fit(model,train_dataloaders=train_loader,val_dataloaders=test_loader)
    
    print("Finished.")

if __name__ == "__main__":
    global PREFIX
    parser = ArgumentParser() 
    parser.add_argument("--fold", default = 1, type = int)
    parser.add_argument("--epoch", default = 50, type = int)
    parser.add_argument("--gpus", default = 2, type = int)
    parser.add_argument("--acce", default = "ddp", type = str)
    parser.add_argument("--val_interval", default = 0.8, type = float)
    parser.add_argument("--profiler", default = "simple", type = str)
    parser.add_argument("--lr", default = 1e-4*5, type = float)
    parser.add_argument("--verbose_step", default = 10, type = int)
    parser.add_argument("--weight_decay", default = 0, type = float)
    parser.add_argument("--hidden", default = 128, type = int)
    parser.add_argument("--heads", default = 2, type = int)
    parser.add_argument("--factor", default = 2, type = int)
    parser.add_argument("--layers", default = 6, type = int)
    parser.add_argument("--batch", default = 32, type = int)
    parser.add_argument("--workers", default = 4, type = int)
    parser.add_argument("--checkpoints", default = None, type = str)
    parser.add_argument("--local_attn", action = "store_true")
    parser.add_argument("--global_attn", action = "store_true")
    parser.add_argument("--value_attn", action = "store_true")
    parser.add_argument("--output", default = None, type = str)
    parser.add_argument("--num_nodes", default = 1, type = int)
    parser.add_argument("--save_memory", action = "store_true")
    parser.add_argument("--ape", action = "store_true")
    parser.add_argument("--kq", default = 8, type = int)
    parser.add_argument("--kv", default = 8, type = int)
    parser.add_argument("--down", default = 2, type = int)
    parser.add_argument("--first", default = 6, type = int)
    parser.add_argument("--prefix", default = ".", type = str)
    
    args = parser.parse_args()
    PREFIX = args.prefix
    main(args)
