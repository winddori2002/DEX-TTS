import numpy as np
import torch
import timm
import torch.nn.functional as F

import torch.nn as nn
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import einops
from itertools import repeat
import collections.abc
import warnings
from functools import partial

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def unpatchify(x, channels=3, n_mels=80):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w      = int(n_mels // patch_size)

    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x

class PatchEmbed2D(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=(80,10000), patch_size=16, stride=16, in_chans=3, embed_dim=768, overlap=True, norm_layer=None, flatten=True):
        super().__init__()

        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride     = to_2tuple(stride)

        self.img_size     = img_size
        self.patch_size   = patch_size
        self.stride       = stride
        self.grid_size    = (img_size[0] // stride[0], img_size[1] // stride[1])
        self.num_patches  = self.grid_size[0] * self.grid_size[1]
        self.padding_size = (patch_size[0] // 2, patch_size[1] // 2)
        self.flatten   = flatten
        self.embed_dim = embed_dim
        if overlap:
            self.proj = nn.Sequential(nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=patch_size, stride=stride, groups=in_chans, padding=self.padding_size),
                                    nn.SiLU(),
                                    nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=1, stride=1)
                                )
        else:
            self.proj = nn.Sequential(nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=patch_size, stride=patch_size, groups=in_chans),
                                    nn.SiLU(),
                                    nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=1, stride=1)
                                    )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def make_conv_pos(e, k, g, is_batch_norm=False):
    pos_conv = nn.Conv2d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())

    return pos_conv

def make_1dconv_pos(e, k, g, is_batch_norm=False):
    pos_conv = nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    pos_conv = nn.Sequential(pos_conv, SamePad1d(k), nn.GELU())

    return pos_conv

class SamePad1d(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, :-self.remove]
        return x

class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, :-self.remove, :-self.remove]
        return x

#################################################################################
#                          Token Masking and Unmasking                          #
#################################################################################

def get_mask(batch, length, mask_ratio, device, mask_type, h, w):

    if mask_type == 'random':
        len_keep    = int(length * (1 - mask_ratio))
        noise       = torch.rand(batch, length, device=device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    elif mask_type == 'freq':
        len_keep    = int(h * (1 - mask_ratio)) * w
        noise       = torch.rand(batch, h, device=device)  # 0, 1, 2, 3,
        ids_shuffle = torch.argsort(noise, dim=1)       
        ids_shuffle = ids_shuffle.repeat_interleave(w).reshape(batch, -1)  # 0, 0, 0, .. 1, 1, 1, ...
        ids_offset  = torch.arange(w, device=device).reshape(1, w).repeat(1, h)
        ids_shuffle = ids_shuffle * w + ids_offset
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
    elif mask_type == 'freq_random':
        len_keep    = int(h * (1 - mask_ratio)) * w
        noise       = torch.rand(batch, h, device=device)  # 0, 1, 2, 3,
        ids_shuffle = torch.argsort(noise, dim=1)       
        ids_shuffle = ids_shuffle.repeat_interleave(w).reshape(batch, -1)  # 0, 0, 0, .. 1, 1, 1, ...
        ids_offset  = torch.arange(w, device=device).reshape(1, w).repeat(1, h)
        ids_shuffle = ids_shuffle * w + ids_offset
        ids_shuffle_a = ids_shuffle[:,:len_keep]
        ids_shuffle_b = ids_shuffle[:,len_keep:]
        ids_shuffle_a = ids_shuffle_a[:,torch.randperm(len_keep)]
        ids_shuffle   = torch.cat([ids_shuffle_a, ids_shuffle_b], 1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
    elif mask_type == 'time':
        len_keep    = int(w * (1 - mask_ratio)) * h
        noise       = torch.rand(batch, w, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_shuffle = ids_shuffle.repeat_interleave(h).reshape(batch, -1)
        ids_offset  = torch.arange(h, device=device).reshape(1, h).repeat(1, w)
        ids_shuffle = ids_shuffle + ids_offset * w
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
    elif mask_type == 'time_random':
        len_keep      = int(w * (1 - mask_ratio)) * h
        noise         = torch.rand(batch, w, device=device)
        ids_shuffle   = torch.argsort(noise, dim=1)
        ids_shuffle   = ids_shuffle.repeat_interleave(h).reshape(batch, -1)
        ids_offset    = torch.arange(h, device=device).reshape(1, h).repeat(1, w)
        ids_shuffle   = ids_shuffle + ids_offset * w
        ids_shuffle_a = ids_shuffle[:,:len_keep]
        ids_shuffle_b = ids_shuffle[:,len_keep:]
        ids_shuffle_a = ids_shuffle_a[:,torch.randperm(len_keep)]
        ids_shuffle   = torch.cat([ids_shuffle_a, ids_shuffle_b], 1)
        ids_restore   = torch.argsort(ids_shuffle, dim=1)
        
    ids_keep = ids_shuffle[:, :len_keep]
    ids_ban  = ids_shuffle[:,len_keep:]

    mask = torch.ones([batch, length], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return {'mask': mask, 
            'ids_keep': ids_keep, 
            'ids_restore': ids_restore,
            'ids_ban': ids_ban}

def mask_out_token(x, ids_keep):
    """
    Mask out the tokens specified by ids_keep.
    Args:
        - x: input sequence, [N, L, D]
        - ids_keep: indices of tokens to keep
    return:
        - x_masked: masked sequence
    """
    N, L, D = x.shape  # batch, length, dim
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    return x_masked


def mask_tokens(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def unmask_tokens(x, ids_restore, mask_token, extras=0):
    # x: [N, T, D] if extras == 0 (i.e., no cls token) else x: [N, T+1, D]
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + extras - x.shape[1], 1)
    x_ = torch.cat([x[:, extras:, :], mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :extras, :], x_], dim=1)  # append cls token
    return x

def merge_tokens(x, x_masked, ids_restore):
    # x: [N, T, D] if extras == 0 (i.e., no cls token) else x: [N, T+1, D]
    x = torch.cat([x_masked, x], dim=1) 
    x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, c_emb_dize, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_emb_dize, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DecoderLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, decoder_hidden_size):
        super().__init__()
        self.norm_decoder = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, decoder_hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_decoder(x), shift, scale)
        x = self.linear(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_emb_size, 2 * final_hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
class DiTMask(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            input_size=(80,10000),
            patch_size=16,
            stride_size=16,
            overlap=True, 
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            in_channels=3,
            out_channels=1,
            conv_pos=16, 
            conv_pos_groups=16,
            mask_type='random',
            use_decoder=False,  # decide if add a lightweight DiT decoder
            norm_layer=partial(nn.LayerNorm, eps=1e-6),  # normalize the encoder output feature
    ):
        super().__init__()
        self.input_size   = input_size
        self.in_channels  = in_channels
        self.out_channels = in_channels
        self.patch_size   = patch_size
        self.num_heads    = num_heads
        self.use_decoder  = use_decoder
        self.overlap      = overlap
        self.stride_size  = stride_size
        self.mask_type    = mask_type
        self.feat_norm    = norm_layer(hidden_size, elementwise_affine=False)

        self.x_embedder = PatchEmbed2D(img_size=input_size, patch_size=patch_size, stride=stride_size, in_chans=in_channels, embed_dim=hidden_size, overlap=overlap, flatten=False)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.freq_new_pos_embed = nn.Parameter(torch.zeros(1, hidden_size, self.x_embedder.grid_size[0], 1))  # | f
        self.pos_conv           = make_conv_pos(hidden_size, conv_pos, conv_pos_groups, is_batch_norm=False)

        self.cls_token      = None
        self.extras         = 0
        self.decoder_extras = 0

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.decoder_layer      = None
        self.decoder_blocks     = None
        self.mask_token         = None
        self.cls_token_embedder = None
        self.enc_feat_embedder  = None
        final_hidden_size       = hidden_size
        if self.use_decoder:
            decoder_hidden_size = hidden_size
            decoder_depth       = depth
            decoder_num_heads   = num_heads

            self.decoder_pos_conv  = make_1dconv_pos(hidden_size, conv_pos, conv_pos_groups, is_batch_norm=False)
            self.decoder_blocks    = nn.ModuleList([
                DiTBlock(decoder_hidden_size, hidden_size, decoder_num_heads, mlp_ratio=mlp_ratio) for _ in
                range(decoder_depth)
            ])
            final_hidden_size = decoder_hidden_size

        self.final_layer = FinalLayer(final_hidden_size, hidden_size, stride_size, self.out_channels)
        self._init_weight()
        
    def _init_weight(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        if self.use_decoder:
            for block in self.decoder_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def get_raw_patch(self, x):
        
        if self.overlap:
            p = self.stride_size
        else:
            p = self.patch_size
        c =  x.shape[1]
        
        x_len   = x.shape[-1]
        if x_len % p != 0:
            s = x_len % p
            x = F.pad(x, (0, p - s))

        h = x.shape[2] // p
        x = x.reshape(shape=(x.shape[0], c, h, p, -1, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(x.shape[0], -1, p**2 * c))
        
        return x
        
    def patchify(self, x):
        
        x_len   = x.shape[-1]
        if x_len % self.patch_size != 0:
            s = x_len % self.patch_size
            x = F.pad(x, (0, self.patch_size - s))

        x = self.x_embedder(x)   # B, Dim, F, T
        h, w = x.shape[2], x.shape[3]

        time_new_pos_embed = self.pos_conv(x)
        time_new_pos_embed = time_new_pos_embed.mean(dim=2, keepdim=True)        
        x = x + time_new_pos_embed[:, :, :, :x.shape[-1]]
        x = x + self.freq_new_pos_embed
        x = x.flatten(2).transpose(1,2)  # (N, T, D), where T = H * W / patch_size ** 2
        
        return x, x_len, h, w

    def unpatchify(self, x):
        channels = self.in_channels
        patch_size = int((x.shape[2] // channels) ** 0.5)
        h = w      = int(self.input_size[0] // patch_size)
        x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
        return x
    
    def forward_encoder(self, x, t):
           
        for block in self.blocks:
            x = block(x, t)  # (N, T, D)
        
        return x
    
    def forward_decoder(self, x, t):
           
        x = x.transpose(1,2)
        decoder_pos_embed = self.decoder_pos_conv(x)
        decoder_pos_embed = decoder_pos_embed.mean(dim=1, keepdim=True)
        x = x + decoder_pos_embed
        x = x.transpose(1,2)

        for block in self.decoder_blocks:
            x = block(x, t)
        
        return x

    def forward(self, x, mask, t, mask_ratio=0, mask_dict=None, feat=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x, x_len, h, w = self.patchify(x) # x: b, t, d
        t              = self.t_embedder(t)

        if self.training and mask_ratio > 0:
            mask_dict = get_mask(x.shape[0], x.shape[1], mask_ratio=mask_ratio, device=x.device, mask_type=self.mask_type, h=h, w=w)
            x         = mask_out_token(x, mask_dict['ids_keep'])
            ids_keep    = mask_dict['ids_keep']
            ids_restore = mask_dict['ids_restore']
        else:
            ids_keep = ids_restore = None
            
        x = self.forward_encoder(x, t)

        if self.training and mask_ratio > 0:
            mask_token = self.mask_token
            if mask_token is None:
                mask_token = torch.zeros(1, 1, x.shape[2]).to(x)  # concat zeros to match shape
            x = unmask_tokens(x, ids_restore, mask_token, extras=self.decoder_extras)
        
        if self.use_decoder:
            x = self.forward_decoder(x, t)

        if not self.use_decoder and (self.training and mask_ratio > 0):
            mask_token = torch.zeros(1, 1, x.shape[2]).to(x)  # concat zeros to match shape
            x = unmask_tokens(x, ids_restore, mask_token, extras=self.extras)
            x = x[:, self.decoder_extras:, :]
        
        x = self.final_layer(x, t)
        x = x[:, self.decoder_extras:, :]  # remove cls token (if necessary)
        x = self.unpatchify(x)             # (N, out_channels, H, W)
        x = x[..., :x_len]
        x = x * mask
            
        return x