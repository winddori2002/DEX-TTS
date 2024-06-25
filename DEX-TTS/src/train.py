import numpy as np
import os
import json
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler 
import torchaudio

from copy import deepcopy
from collections import OrderedDict

from src.dataset import *
from src.utils import *
from src.evaluation import *
from model import DeXTTS

from tqdm import tqdm

def get_mask_ratio_fn(name='constant', ratio_scale=0.5, ratio_min=0.0):
    if name == 'cosine2':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 2 + ratio_min
    elif name == 'cosine3':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 3 + ratio_min
    elif name == 'cosine4':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 4 + ratio_min
    elif name == 'cosine5':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 5 + ratio_min
    elif name == 'cosine6':
        return lambda x: (ratio_scale - ratio_min) * np.cos(np.pi * x / 2) ** 6 + ratio_min
    elif name == 'exp':
        return lambda x: (ratio_scale - ratio_min) * np.exp(-x * 7) + ratio_min
    elif name == 'linear':
        return lambda x: (ratio_scale - ratio_min) * x + ratio_min
    elif name == 'constant':
        return lambda x: ratio_scale
    elif name == 'random':
        return lambda x: np.random.uniform(ratio_min, ratio_scale)
    else:
        raise ValueError('Unknown mask ratio function: {}'.format(name))

@torch.no_grad()
def update_ema(ema_model, model, decay=0.99999):
    """
    Step the EMA model towards the current model.
    """
    ema_params   = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

class Trainer:

    def __init__(self, data, cfg):

        self.cfg       = cfg
        self.model     = DeXTTS(cfg.model).to(cfg.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(cfg.train.lr))
        self.scaler    = GradScaler(enabled=cfg.train.amp)
        
        self.train_loader = data['train']
        self.val_loader   = data['valid']
        self.tester       = Tester(cfg)
        
        self.mask_ratio_fn = get_mask_ratio_fn(name='random',ratio_scale=cfg.train.mask_ratio)
        self.cur_step      = 0
        self.total_step    = len(self.train_loader) * self.cfg.train.epoch

        # Write param size & [model, conv_module, config files]
        param_size          = count_parameters(self.model)
        self.cfg.param_size = np.round(param_size/1000000,2)
        print(f'Param size: {cfg.param_size}M')
        
        # compile model and ema model
        self.ema = deepcopy(self.model).to(cfg.device)  # Create an EMA of the model for use after training
        requires_grad(self.ema, False)
        update_ema(self.ema, self.model, decay=0)
        self.ema.eval()

        # logging
        if cfg.logging:
            print('---logging start---')
            neptune_load(get_cfg_params(cfg))
            
        # checkpoint
        if cfg.resume is not None:
            self._resume_checkpoint()
    
    def _save_log(self, msg):
        with open(f'{self.cfg.checkpoint}/log.txt', 'a') as f:
            f.write(msg)
                
    def _resume_checkpoint(self):
        checkpoint = torch.load(f'{self.cfg.checkpoint}/model-last.pth', map_location=self.cfg.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.ema.load_state_dict(checkpoint['ema'])
        self.optimizer.load_state_dict(checkpoint['optimizer']) 
        print('---load previous weigths and optimizer for resume training---')
        
    def _save_checkpoint(self, scores, epoch, phase='train', opt='best'):
        checkpoint = {'scores':     scores,
                      'state_dict': self.model.state_dict(),
                      'ema':        self.ema.state_dict(),
                      'optimizer':  self.optimizer.state_dict()}   
        if opt=='best':
            torch.save(checkpoint, f'{self.cfg.checkpoint}/model-{phase}-best.pth')
        elif opt=='last':
            torch.save(checkpoint, f'{self.cfg.checkpoint}/model-last.pth')           
        else:
            torch.save(checkpoint, f'{self.cfg.checkpoint}/model-{epoch}.pth')

    def train(self):
        
        best_train_loss = 1000000
        best_val_loss   = 1000000
        for epoch in range(1, self.cfg.train.epoch+1):
            
            self.model.train()
            train_dur_loss, train_diff_loss, train_prior_loss, train_vq_loss = self._run_epoch(self.train_loader)             
            train_loss = (train_dur_loss + train_diff_loss + train_prior_loss + train_vq_loss) / 4

            self.model.eval()
            with torch.no_grad():
                val_dur_loss, val_diff_loss, val_prior_loss, val_vq_loss = self._run_epoch(self.val_loader, valid=True)
                val_loss  = (val_dur_loss + val_diff_loss + val_prior_loss + val_vq_loss) / 4 

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                self._save_checkpoint([best_train_loss], epoch, phase='train', opt='best')   
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint([best_val_loss], epoch, phase='val', opt='best')      

            if epoch % self.cfg.train.save_epoch == 0:
                self._save_checkpoint([best_train_loss], epoch, opt='epoch')               
            
            self._save_checkpoint([best_train_loss], epoch, opt='last')
            
            
            step = epoch * (len(self.train_loader.dataset) // self.cfg.train.batch_size)
            msg  = "Epoch: {:03d} | Step: {:03d} | trn loss: {:.4f} | dur loss: {:.4f} | diff loss: {:.4f} | prior loss: {:.4f} | vq loss: {:.4f}\n".format(epoch, step, 
                                                                                                                                         train_loss, train_dur_loss, 
                                                                                                                                         train_diff_loss, train_prior_loss, train_vq_loss)
            msg  += "Epoch: {:03d} | Step: {:03d} | val loss: {:.4f} | dur loss: {:.4f} | diff loss: {:.4f} | prior loss: {:.4f} | vq loss: {:.4f} \n".format(epoch, step, 
                                                                                                                                         val_loss, val_dur_loss, 
                                                                                                                                         val_diff_loss, val_prior_loss, val_vq_loss)
            print(msg)
            self._save_log(msg+'\n\n')
            
            if self.cfg.logging == True:
                neptune.log_metric('cur epoch', epoch)
                neptune.log_metric('train loss', train_loss)
                neptune.log_metric('train dur loss', train_dur_loss)
                neptune.log_metric('train diff loss', train_diff_loss)
                neptune.log_metric('train prior loss', train_prior_loss)
                neptune.log_metric('train vq loss', train_vq_loss)
                neptune.log_metric('val loss', val_loss)
                neptune.log_metric('val dur loss', val_dur_loss)
                neptune.log_metric('val diff loss', val_diff_loss)
                neptune.log_metric('val prior loss', val_prior_loss)
                neptune.log_metric('val vq loss', val_vq_loss)
                
            if epoch % self.cfg.train.syn_every == 0:
                print('--- Synthesize samples ---')
                self.tester.synthesize()
                
    def _run_epoch(self, data_loader, valid=False):
        
        total_dur_loss   = 0
        total_prior_loss = 0
        total_diff_loss  = 0
        total_vq_loss    = 0
        for i, batch in enumerate(tqdm(data_loader)):

            x, x_lengths     = batch['x'].to(self.cfg.device), batch['x_lengths'].to(self.cfg.device)
            y, y_lengths     = batch['y'].to(self.cfg.device), batch['y_lengths'].to(self.cfg.device)
            ref, ref_lengths = batch['ref'].to(self.cfg.device), batch['ref_lengths'].to(self.cfg.device)
            sty, sty_lengths = batch['sty'].to(self.cfg.device), batch['sty_lengths'].to(self.cfg.device)
            lf0, lf0_lengths = batch['lf0'].to(self.cfg.device), batch['lf0_lengths'].to(self.cfg.device)
            spk              = batch['spk'].to(self.cfg.device)
            
            curr_mask_ratio = 0 #self.mask_ratio_fn(self.cur_step / self.total_step)  
            with autocast(enabled=self.cfg.train.amp):
                dur_loss, prior_loss, diff_loss, vq_loss = self.model.compute_loss(x, x_lengths, y, y_lengths, ref, ref_lengths, sty, sty_lengths, lf0, lf0_lengths, spk=None, mask_ratio=curr_mask_ratio, out_size=self.cfg.train.out_size)
                loss = sum([dur_loss, prior_loss, diff_loss, vq_loss])
            
                if not valid:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.train.max_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    update_ema(self.ema, self.model)
                
            total_dur_loss   += dur_loss.item()
            total_diff_loss  += diff_loss.item()
            total_prior_loss += prior_loss.item() 
            total_vq_loss    += vq_loss.item() 
        
        return total_dur_loss/(i+1), total_diff_loss/(i+1), total_prior_loss/(i+1), total_vq_loss/(i+1)
    