""" from https://github.com/jaywalnut310/glow-tts """

import math

import torch
import torch.nn as nn

from model.utils import sequence_mask, convert_pad_shape
from model.retnet import *
from model.retnet_cfg import *


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean)**2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, 
                 n_layers, p_dropout):
        super(ConvReluNorm, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, 
                                                kernel_size, padding=kernel_size//2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super(DurationPredictor, self).__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, 
                                      kernel_size, padding=kernel_size//2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, 
                                      kernel_size, padding=kernel_size//2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj   = torch.nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask

class TextEncoder(nn.Module):
    def __init__(self, n_vocab, n_feats, n_channels, filter_channels, filter_channels_dp, n_heads, n_layers, kernel_size, p_dropout, use_softmax, use_decay, window_size=None, spk_emb_dim=64, n_spks=1):
        super(TextEncoder, self).__init__()
        self.n_vocab = n_vocab
        self.n_feats = n_feats
        self.n_channels = n_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks

        self.emb = torch.nn.Embedding(n_vocab, n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, n_channels**-0.5)

        self.prenet = ConvReluNorm(n_channels, n_channels, n_channels, 
                                   kernel_size=5, n_layers=3, p_dropout=0.5)
        
        CONFIG = RetNetConfig(decoder_layers=n_layers,
                            decoder_embed_dim=n_channels + (spk_emb_dim if n_spks > 1 else 0),
                            decoder_value_embed_dim=n_channels + (spk_emb_dim if n_spks > 1 else 0),
                            decoder_retention_heads=n_heads,
                            decoder_ffn_embed_dim=filter_channels,
                            dropout=p_dropout,
                            )

        # self.encoder = RetNet(n_layers, n_channels + (spk_emb_dim if n_spks > 1 else 0), filter_channels, n_heads)
        self.encoder = RetNetModel(CONFIG, tensor_parallel=False, use_softmax=use_softmax, use_decay=use_decay)
        self.proj_m  = torch.nn.Conv1d(n_channels + (spk_emb_dim if n_spks > 1 else 0), n_feats, 1)
        self.proj_w  = DurationPredictor(n_channels + (spk_emb_dim if n_spks > 1 else 0), filter_channels_dp, 
                                        kernel_size, p_dropout)

    def forward(self, x, x_lengths, spk=None):
        x = self.emb(x) * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1)
        x_mask    = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        x = self.prenet(x, x_mask)
        if self.n_spks > 1:
            x = torch.cat([x, spk.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        x = self.encoder(inputs_embeds=x.transpose(1,2), attention_mask=x_mask)[0].transpose(1,2) * x_mask
        mu = self.proj_m(x) * x_mask

        x_dp = torch.detach(x)
        logw = self.proj_w(x_dp, x_mask)

        return mu, logw, x_mask