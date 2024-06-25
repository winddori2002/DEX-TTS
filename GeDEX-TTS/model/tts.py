import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# import monotonic_align
from model import monotonic_align
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility


class GeDEXTTS(nn.Module):
    def __init__(self, cfg):
        super(GeDEXTTS, self).__init__()
        
        self.n_spks  = cfg.n_spks
        self.n_feats = cfg.n_feats
        
        if cfg.n_spks > 1:
            self.spk_emb = torch.nn.Embedding(cfg.n_spks, cfg.spk_emb_dim)
        self.encoder = TextEncoder(**cfg.encoder, n_vocab=cfg.n_vocab, n_feats=cfg.n_feats, n_spks=cfg.n_spks, spk_emb_dim=cfg.spk_emb_dim)
        self.decoder = Diffusion(**cfg.decoder, dit_cfg=cfg.dit, n_feats=cfg.n_feats, n_spks=cfg.n_spks, spk_emb_dim=cfg.spk_emb_dim)

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, spk=None, length_scale=1.0):

        if self.n_spks > 1:
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk=spk)

        w             = torch.exp(logw) * x_mask
        w_ceil        = torch.ceil(w) * length_scale
        y_lengths     = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length  = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask    = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn      = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y    = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y    = mu_y.transpose(1, 2)
        enc_out = mu_y[:, :, :y_max_length]
        
        # Generate sample by performing reverse dynamics
        dec_out = self.decoder(mu_y, y_mask, mu_y, temperature=temperature, n_timesteps=n_timesteps, spk=spk, infer=True)
        dec_out = dec_out[:, :, :y_max_length]

        return enc_out, dec_out, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, out_size=None, mask_ratio=0):

        if self.n_spks > 1:
            spk = self.spk_emb(spk)
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk=spk)
        y_max_length       = y.shape[-1]

        y_mask    = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad(): 
            const       = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor      = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square    = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square   = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior   = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_    = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            
            if out_size < y.shape[-1]: 
                max_offset = (y_lengths - out_size).clamp(0)
                offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
                out_offset = torch.LongTensor([
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]).to(y_lengths)
                
                attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
                y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
                y_cut_lengths = []
                for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                    y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                    y_cut_lengths.append(y_cut_length)
                    cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                    y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                    attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
                y_cut_lengths = torch.LongTensor(y_cut_lengths)
                y_cut_mask    = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
                
                attn   = attn_cut
                y      = y_cut
                y_mask = y_cut_mask
            
        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of score-based decoder
        diff_loss  = self.decoder(y, y_mask, mu_y, spk=spk, infer=False, mask_ratio=mask_ratio)
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        return dur_loss, prior_loss, diff_loss