import random
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F


class Augment(nn.Module):

    """Shift."""

    def __init__(self,  freq_mask_num=1, time_mask_num=1, freq_mask=False, time_mask=True):
        """
        :param shift: randomly shifts the signals up to a given factor
        :param same: shifts both clean and noisy files by the same factor
        """
        super().__init__()
  
        self.freq_mask_num  = freq_mask_num
        self.time_mask_num  = time_mask_num
        self.freq_mask      = freq_mask
        self.time_mask      = time_mask 
        
    def freq_mask_augment(self, x, freq_mask_para):
        
        v, tau = x.shape

        for i in range(self.freq_mask_num):
            f  = np.random.uniform(low=0.0, high=freq_mask_para)
            f  = int(f)
            f0 = random.randint(0, v-f)
            x[f0:f0+f, :] = 0

        return x
    
    def time_mask_augment(self, x, time_mask_para):
        
        v, tau = x.shape

        for i in range(self.time_mask_num):
            t  = np.random.uniform(low=0.0, high=time_mask_para)
            t  = int(t)
            t0 = random.randint(0, tau - t)
            x[:, t0:t0 + t]  = 0

        return x
    
    def shift_augment(self, x):
        
        c, t = x.shape
        idx  = int(np.random.uniform(t))
        x    = torch.cat([x[:,idx:], x[:,:idx]], dim=1)
        
        return x

    def forward(self, x, aug_type, time_mask_para=27, freq_mask_para=30):
        
        # x: [M, T]
        
        if len(x.size()) != 2:
            x = x.unsqueeze(0)
            
        if 'T' in aug_type:
            x = self.time_mask_augment(x, time_mask_para)
            
        elif 'F' in aug_type:
            x = self.freq_mask_augment(x, freq_mask_para)

        elif 'S' in aug_type:
            x = self.shift_augment(x)

        return x.squeeze(1)
        