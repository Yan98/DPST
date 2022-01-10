#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from model import Mass2Protein
from torch.utils.data import DataLoader
import torch_optimizer as optim
import torch
import torch.nn as nn
from typing import Optional, Sequence
from torch.nn import functional as F
import time
import datetime
import subprocess 
import pytorch_lightning as pl
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import os

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            #c = x.shape[1]
            #x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            d = x.shape[-1]
            x = x.view(-1,d)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.
        
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

def accuracy(y,y_pred,mask):
    y_pred = y_pred.argmax(2).long()
    acc    = ((y == y_pred)[mask]).sum() / (mask.sum() + 1e-8)
    return acc#.item()
    

def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.
    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.
    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl   

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input  = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = [i.cuda(non_blocking=True) for i in self.next_input]
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input


class DenovoTrainer:
    def __init__(self,model:Mass2Protein, train_loader:DataLoader, test_loader:DataLoader,
                 lr = 1e-3, with_cuda = True, logfun = print):
        self.logfun = logfun
        self.cuda_condition = torch.cuda.is_available() and with_cuda
        self.train_loader = train_loader
        self.test_loader  = test_loader
        
        self.model     = model
        self.criterion = focal_loss(gamma = 2, ignore_index = test_loader.dataset.vocab.pad_index)
        if self.cuda_condition:
            self.model = model.cuda()
            self.criterion = self.criterion.cuda()
        #self.optim        = optim.AdaBelief(
        #                    self.model.parameters(),
        #                    lr= lr,
        #                    betas=(0.9, 0.999),
        #                    eps=1e-3,
        #                    weight_decay=0,
        #                    amsgrad=False,
        #                   weight_decouple=False,
        #                    fixed_decay=False,
        #                   rectify=False,
        #                )
        self.optim        = torch.optim.Adam(
                            self.model.parameters(),
                            lr = lr,
                            betas = (0.9, 0.999), 
                            #weight_decay = 0.01
            )
        #self.optim_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min', factor=0.5, threshold=1e-4, cooldown=10, min_lr=1e-5)
        
        self.logfun("Total Parameters: " + str(sum([p.nelement() for p in self.model.parameters()])))
        
    def train(self,epochs, verbose_step = 10, evaluation_step = 1000):
        
        start_time = time.time()
        pad_index  = self.test_loader.dataset.vocab.pad_index
        train_loss = 0
        min_loss   = float("inf")
        max_acc    = float("-inf")
        max_eval_acc  = float("-inf")
        min_eval_loss = float("inf")
        flag = True
        for epoch in range(epochs):
            #pref = data_prefetcher(self.train_loader)
            #data = pref.next()
            #i    = 0
            #while data is not None:
            #    specturm,peptides,lengths = data
            
            #if epoch > 5 and flag:
            #    for g in self.optim.param_groups:
            #        g['lr'] = 1e-3
            #        flag    = False
            
            for i,data in enumerate(self.train_loader):
                if self.cuda_condition:
                    data = {k : v.cuda() for k,v in data.items()}
                self.optim.zero_grad()
                predicted_peptides = self.model(data)
                loss     = self.criterion(predicted_peptides,data["aa_target"])
                #loss = self.criterion(predicted_peptides.reshape(predicted_peptides.size(0) * predicted_peptides.size(1),-1),peptides.reshape(-1))
                loss.backward()
                self.optim.step()
                
                train_loss = loss.item()
                train_acc  = accuracy(data["aa_target"], predicted_peptides, data["aa_mask"])
                max_acc    = max(max_acc,  train_acc)
                min_loss   = min(min_loss, train_loss)
                
                if i % verbose_step == 0:
                    
                    batches_done = epoch  * len(self.train_loader) + i + 1
                    batches_left = epochs * len(self.train_loader) - batches_done
                    time_left    = datetime.timedelta(seconds = batches_left * (time.time() - start_time) / batches_done)
                    
                    self.logfun(
                        "[Epoch %d/%d] [Batch %d/%d] [Loss: %f, Acc: %f] [Min Loss: %f, Max Acc: %f] ETA: %s" % 
                        (epoch,
                         epochs,
                         i,
                         len(self.train_loader),
                         train_loss,
                         train_acc,
                         min_loss,
                         max_acc,
                         time_left
                            )
                        
                        )
                #if self.cuda_condition:
                #    for line in subprocess.check_output(["nvidia-smi"]).decode("utf-8").split("\n"):
                #        self.logfun(line)
                if i % evaluation_step == 0:
                    if self.cuda_condition:
                        for line in subprocess.check_output(["nvidia-smi"]).decode("utf-8").split("\n"):
                            self.logfun(line)

                if i % evaluation_step == 0 and i != 0:
                    with torch.no_grad():
                        self.model.eval()
                        torch.cuda.empty_cache()
                        self.logfun("Start Evaluation")
                        total_loss = 0
                        correct    = 0
                        num        = 0
                        for j,data in enumerate(self.test_loader):
                            if self.cuda_condition:
                                data = {k : v.cuda() for k,v in data.items()}
                            
                            predicted_peptides = self.model(data)
                            loss     = self.criterion(predicted_peptides,data["aa_target"])
                            #loss = self.criterion(predicted_peptides.reshape(predicted_peptides.size(0) * predicted_peptides.size(1),-1),peptides.reshape(-1))
                            total_loss += loss.item()
                            correct    += accuracy(data["aa_target"], predicted_peptides, data["aa_mask"]) * (data["aa_mask"].sum().item())
                            num        += (data["aa_mask"].sum().item())
                        
                        total_loss = total_loss / len(self.test_loader)
                        acc        = correct / num
                        
                        if acc > max_eval_acc:
                            self.save(epoch, acc)                    
                        max_eval_acc = max(max_eval_acc,acc)
                        min_eval_loss = min(min_eval_loss, total_loss)
                        
                        self.logfun("==" * 25)
                        self.logfun(
                            "[Acc :%f, Loss: %f] [Min Loss :%f, Max Acc: %f]" %
                            (acc,
                             total_loss,
                             min_eval_loss,
                             max_eval_acc
                                )
                            )
                        self.logfun("==" * 25)
                        self.logfun("End Evaluation")
                        
                        self.model.train()
                
                #data = pref.next()
                #i   += 1 
                    
    def save(self, epoch, acc, file_path="output/dn"):
        output_path = file_path + "_%d_%d.pt" % (epoch,acc)
        torch.save(self.model.state_dict(), output_path)
        self.logfun("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


class TrainerModel(pl.LightningModule):
    
    def __init__(self, config,  model):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = focal_loss(gamma = 2, ignore_index = config.pad_index)
        self.automatic_optimization = False
        self.min_loss   = float("inf")
        self.max_acc    = float("-inf")
        self.max_eval_acc  = float("-inf")
        self.min_eval_loss = float("inf")
        self.start_time  = None
        
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes) * self.trainer.num_nodes
        return len(dataset) // num_devices
        
    def training_step(self,data,idx):
        
        if self.current_epoch == 0 and idx == 0:
            self.start_time  = time.time()
        
        optimizer = self.optimizers()
        
        predicted_peptides = self.model(data)
        loss   = self.criterion(predicted_peptides,data["aa_target"])
        train_acc  = accuracy(data["aa_target"], predicted_peptides, data["aa_mask"])
        
        #return loss
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        
        self.produce_log(loss.detach(),train_acc.detach(),data,idx)
        
        
    def produce_log(self,loss,acc,data,idx):
        
        train_loss = self.all_gather(loss).mean().item()
        train_acc = self.all_gather(acc).mean().item()
        
        self.max_acc  = max(self.max_acc,  train_acc)
        self.min_loss   = min(self.min_loss, train_loss)
        
        if self.trainer.is_global_zero and loss.device.index == 0 and idx % self.config.verbose_step == 0:
            
            current_lr = self.optimizers().param_groups[0]['lr']
            
            len_loader = self.num_training_steps
            
            batches_done = self.current_epoch  * len_loader + idx + 1
            batches_left = self.trainer.max_epochs * len_loader - batches_done
            time_left    = datetime.timedelta(seconds = batches_left * (time.time() - self.start_time) / batches_done)
                    
            self.config.logfun(
                        "[Epoch %d/%d] [Batch %d/%d] [Loss: %f, Acc: %f, lr: %f] [Min Loss: %f, Max Acc: %f] ETA: %s" % 
                        (self.current_epoch,
                         self.trainer.max_epochs,
                         idx,
                         len_loader,
                         train_loss,
                         train_acc,
                         current_lr * 1e3,
                         self.min_loss,
                         self.max_acc,
                         time_left
                            )
                        
                        )
            

        
    def validation_step(self,data,idx):
        
        predicted_peptides = self.model(data)
        loss     = self.criterion(predicted_peptides,data["aa_target"])
        
        acc    = accuracy(data["aa_target"], predicted_peptides, data["aa_mask"]) * (data["aa_mask"].sum())
        num    = data["aa_mask"].sum()
        
        return loss, acc,num
        
    def validation_epoch_end(self,outputs):
        
        logfun = self.config.logfun
        
        total_loss = sum([i[0] for i in outputs])
        correct = sum(i[1] for i in outputs)
        num = sum(i[2] for i in outputs)
        
        total_loss = torch.mean(self.all_gather(total_loss)).item()
        correct  = torch.sum(self.all_gather(correct)).item()
        num  = torch.sum(self.all_gather(num)).item()
        
        
        if self.trainer.is_global_zero and self.trainer.num_gpus != 0:
            for line in subprocess.check_output(["nvidia-smi"]).decode("utf-8").split("\n"):
                self.config.logfun(line)
            
            total_loss = total_loss / len(outputs)
            acc        = correct / num
            if acc > self.max_eval_acc:
                 self.save(self.current_epoch, total_loss,acc)                    
            self.max_eval_acc = max(self.max_eval_acc,acc)
            self.min_eval_loss = min(self.min_eval_loss, total_loss)
                            
            logfun("==" * 25)
            logfun(
                "[Acc :%f, Loss: %f] [Min Loss :%f, Max Acc: %f]" %
                (acc,
                 total_loss,
                 self.min_eval_loss,
                 self.max_eval_acc
                 )
                )            
            

            logfun("==" * 25)
            logfun("End Evaluation")
        
    def save(self, epoch,loss, acc):
        output_path = os.path.join(self.config.store_dir, "%d_%f_%f.pt" % (epoch,loss,acc))
        torch.save(self.model.state_dict(), output_path)
        self.config.logfun("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
                      
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                            self.parameters(),
                            lr = self.config.lr,
                            betas = (0.9, 0.999),
                            weight_decay = self.config.weight_decay,
            )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.num_training_steps//2,
            cycle_mult=1.0,
            max_lr=self.config.lr,
            min_lr=self.config.lr/5,
            warmup_steps= 100
        )
        
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}                       
                      
                        
                        
                        
        
     
