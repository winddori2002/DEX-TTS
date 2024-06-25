import math
import torch
from einops import rearrange

from model.base import *
from model.dit import DiTMask
from model.ref_encoder import *
from model.edm import *
from model.utils import sequence_mask

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(nn.Module):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionDenoiser(nn.Module):
    def __init__(self, dim, dit_cfg, model_type='vit', dim_mults=(1, 2, 4), groups=8, n_spks=None, spk_emb_dim=64, n_feats=80, pe_scale=1000):
        super(DiffusionDenoiser, self).__init__()
        self.dim         = dim
        self.dim_mults   = dim_mults
        self.groups      = groups
        self.n_spks      = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale    = pe_scale

        self.norm         = InstanceNorm1D()
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp          = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(), torch.nn.Linear(dim * 4, dim))
        self.mlp_adap     = torch.nn.Sequential(torch.nn.Linear(dim, dim), Mish(), torch.nn.Linear(dim, dim * 2))
        self.mlp_adap_sty = torch.nn.Sequential(torch.nn.Linear(dim, dim), Mish(), torch.nn.Linear(dim, dim * 2))
        
        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                                               torch.nn.Linear(spk_emb_dim * 4, n_feats))
            
        dims       = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out     = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups   = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim              = dims[-1]
        dit_cfg.in_channels  = mid_dim
        dit_cfg.out_channels = mid_dim
        
        self.tv_adaptor  = TVAdaptor(mid_dim)
        self.tiv_adaptor = TIVAdaptor(mid_dim)
        
        h        = int(n_feats/(2**(len(dim_mults)-1)))
        img_size = (h, 1000)
        if model_type == 'dit':
            self.vit = DiTMask(**dit_cfg, input_size=img_size)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                     ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)
            
    def _stack_stats(self, trg_skips, trg_lengths):
        
        trg_means = []
        trg_stds  = []
        for trg in trg_skips:
            m, s = self.norm.cal_stats(trg, trg_lengths)
            trg_means.append(m) # [B, C, 1]
            trg_stds.append(s)
        trg_means = torch.cat(trg_means, dim=-1).transpose(1,2) # [(B,C,1);(B,C,1)] -> (B,L,C)
        trg_stds  = torch.cat(trg_stds, dim=-1).transpose(1,2)

        return trg_means, trg_stds

    def forward(self, x, mask, mu, t, ref, ref_lengths, sty, sty_lengths, spk=None, mask_ratio=0):
        
        sty_mask  = torch.unsqueeze(sequence_mask(sty_lengths, sty.size(2)), 1).to(sty.dtype)

        # get ref stats 
        ref_mean, ref_std = self._stack_stats(ref, ref_lengths)
        ref      = (ref_mean, ref_std)
        
        x = torch.stack([mu, x], 1)
        
        t_init = self.time_pos_emb(t, scale=self.pe_scale)
        t_unet = self.mlp(t_init)
        t_adap = self.mlp_adap(t_init).unsqueeze(1)
        t_adap_sty = self.mlp_adap_sty(t_init).unsqueeze(-1)
        
        mask   = mask.unsqueeze(1)
        
        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t_unet)
            x = resnet2(x, mask_down, t_unet)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks    = masks[:-1]
        mask_mid = masks[-1]
        x        = self.tv_adaptor(x, mask_mid, sty, sty_mask.unsqueeze(1), t_adap_sty)
        x        = self.tiv_adaptor(x, ref, ref_lengths, t_adap)        
        x        = self.vit(x, mask_mid, t, mask_ratio=mask_ratio)
        

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t_unet)
            x = resnet2(x, mask_up, t_unet)
            x = attn(x)
            x = upsample(x * mask_up)

        x      = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)

class Diffusion(nn.Module):
    def __init__(self, n_feats, dim, dit_cfg, loss_type='base', precond='edm', model_type='vit', dim_mults=(1, 2), n_spks=1, spk_emb_dim=64, pe_scale=1000):
        super().__init__()
        
        self.denoise_fn    = DiffusionDenoiser(dim, dit_cfg, model_type=model_type, dim_mults=dim_mults, n_spks=n_spks, spk_emb_dim=spk_emb_dim, pe_scale=pe_scale)
        self.precond_model = EDMPrecond(self.denoise_fn)
        self.loss_fn       = EDMLoss(n_feats=n_feats, loss_type=loss_type)
        self.sampler       = lambda z, mask, mu, ref, ref_lengths, sty, sty_lengths, spk, steps: ablation_sampler(net=self.precond_model, latents=z, mask=mask, mu=mu, ref=ref, ref_lengths=ref_lengths, sty=sty, sty_lengths=sty_lengths, spk=spk, num_steps=steps, solver='euler', discretization='edm', schedule='linear', scaling='none')


        print(f'Model:{model_type}, Precond & Sampler: {precond}')
        
    def forward(self, x, mask, mu, ref, ref_lengths, sty, sty_lengths, n_timesteps=1, spk=None, infer=False, temperature=1.0, mask_ratio=0):

        if not infer:
            loss = self.loss_fn(self.precond_model, x, mask, mu, ref, ref_lengths, sty, sty_lengths, spk=spk, mask_ratio=mask_ratio)
            return loss       
        else:
            shape = (mu.shape[0], 80, mu.shape[2])
            x = torch.randn(shape, device=x.device) / temperature + mu
            x = self.sampler(x, mask, mu, ref, ref_lengths, sty, sty_lengths, spk, n_timesteps)
            return x
