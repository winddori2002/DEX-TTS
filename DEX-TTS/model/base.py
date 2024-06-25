import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, norm=True, norm_type='bn', bias=False):
        super().__init__()

        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.relu = nn.ReLU() if relu else None
        
        if norm:
            if norm_type == 'bn':
                self.bn = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.01, affine=True)
                self.ln = None
            elif norm_type == 'ln':
                self.bn = None
                self.ln = nn.LayerNorm(out_channels, eps=1e-5)
        else:
            self.bn = None
            self.ln = None
        
    def forward(self, x):
        
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.ln is not None:
            x = self.ln(x.transpose(1,2))
            x = x.transpose(1,2)
    
        return x
    
class InstanceNorm1D(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def cal_stats(self, x, x_lengths=None):
        
        # input: b, c, t
        mean = x.mean(-1).unsqueeze(-1)
        std  = (x.var(-1) + self.eps).sqrt().unsqueeze(-1)
        
        return mean, std

    def forward(self, x, x_lengths=None, return_stats=False):
        
        mean, std = self.cal_stats(x)
        x               = (x - mean) / std

        if return_stats:
            return x, mean, std
        else:
            return x
        
class InstanceNorm2D(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def cal_stats(self, x):
        
        size = x.size()
        assert (len(size) == 4)
        N, C = size[:2]
        x_var = x.view(N, C, -1).var(dim=2) + self.eps
        x_std = x_var.sqrt().view(N, C, 1, 1)
        x_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return x_mean, x_std

    def forward(self, x, return_stats=False):
        
        size      = x.size()
        mean, std = self.cal_stats(x)
        x         = (x - mean.expand(size)) / std.expand(size)

        if return_stats:
            return x, mean, std
        else:
            return x
        
class FilteredInstanceNorm1D(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.masked_mean = lambda x, x_lens: x[:,:x_lens].mean(-1).unsqueeze(-1)
        self.masked_std  = lambda x, x_lens: (x[:,:x_lens].var(-1) + self.eps).sqrt().unsqueeze(-1)

    def cal_stats(self, x, x_lens):
               
        mean = torch.stack(list(map(self.masked_mean, x, x_lens)))
        std  = torch.stack(list(map(self.masked_std, x, x_lens)))
        return mean, std

    def forward(self, x, x_lens, return_stats=False):
        
        mean, std = self.cal_stats(x, x_lens)
        x         = (x - mean) / std

        if return_stats:
            return x, mean, std
        else:
            return x
        
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        
        # b, c, t
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)  # b, 1, t
        variance = torch.mean((x - mean)**2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
    
class AdaptiveLayerNorm(nn.Module):

    def __init__(self,
                hidden_size,
                epsilon=1e-5
                ):
        super(AdaptiveLayerNorm, self).__init__()

        self.epsilon = epsilon
        self.W_scale = nn.Linear(hidden_size, hidden_size)
        self.W_bias  = nn.Linear(hidden_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)
    
    def forward(self, x, sty):

        # x: b, t, c / sty: b, c
        
        mean = x.mean(dim=-1, keepdim=True)
        var  = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std  = (var + self.epsilon).sqrt()
        y    = (x - mean) / std
        
        scale = self.W_scale(sty)
        bias  = self.W_bias(sty)
        y *= scale.unsqueeze(1)
        y += bias.unsqueeze(1)

        return y