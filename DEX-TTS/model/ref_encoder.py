import math
import torch
from torch import nn
from torch.nn import functional as F
from model.base import LayerNorm, BasicConv, InstanceNorm1D, InstanceNorm2D


class Projection(nn.Module):
    def __init__(self, c_in, c_h, kernel_size, p_drop=0.1):
        super(Projection, self).__init__()
        self.in_channels = c_in
        self.filter_channels = c_h
        self.p_dropout = p_drop

        self.drop = torch.nn.Dropout(p_drop)
        self.conv_1 = torch.nn.Conv1d(c_in, c_h, 
                                      kernel_size, padding=kernel_size//2)
        self.norm_1 = LayerNorm(c_h)
        self.conv_2 = torch.nn.Conv1d(c_h, c_h, 
                                      kernel_size, padding=kernel_size//2)
        self.norm_2 = LayerNorm(c_h)
        self.proj   = torch.nn.Conv1d(c_h, c_h, 1)

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

class LF0Encoder(nn.Module):
    def __init__(self, c_h, c_out, c_out_g, num_layer, c_in=1):
        super().__init__()

        self.in_conv   = BasicConv(c_in, c_h, kernel_size=3, stride=1, padding=1, relu=True, norm=True, norm_type='ln')
        self.rnn_layer = nn.GRU(c_h, c_h//2, num_layer, batch_first=True, bidirectional=True)
        self.out_conv  = BasicConv(c_h, c_out, kernel_size=3, stride=1, padding=1, relu=True, norm=True, norm_type='ln')
        self.proj      = Projection(c_out, c_out_g, kernel_size=3, p_drop=0.1)
        
    def forward(self, lf0, mask):
        # mask: b, 1, t
        lf0    = lf0.unsqueeze(1) # b, c, t
        lf0    = self.in_conv(lf0 * mask) * mask    # B, C, T
        lf0, _ = self.rnn_layer(lf0.transpose(1,2)) # B, T, C
        lf0    = self.out_conv(lf0.transpose(1,2) * mask) * mask  # B, C, T 

        lf0_dec = lf0.detach()
        lf0_dec = self.proj(lf0_dec, mask) 

        return lf0, lf0_dec
        
class TIVEncoderBlock(nn.Module):
    def __init__(self, c_in, c_h):
        super().__init__()
        
        self.conv_block = nn.Sequential(BasicConv(c_in, c_h, kernel_size=3, stride=1, padding=1, relu=True, norm=True, norm_type='bn'),
                                        BasicConv(c_h, c_in, kernel_size=3, stride=1, padding=1, relu=False, norm=False))
        
    def forward(self, x):

        x = x + self.conv_block(x)

        return x
    
class TVEncoderBlock(nn.Module):
    def __init__(self, c_in, c_h):
        super().__init__()
        
        self.conv_block = nn.Sequential(BasicConv(c_in, c_h, kernel_size=3, stride=1, padding=1, relu=True, norm=True, norm_type='ln'),
                                        BasicConv(c_h, c_in, kernel_size=3, stride=1, padding=1, relu=False, norm=False))
        
    def forward(self, x):

        x = x + self.conv_block(x)

        return x

class TIVEncoder(nn.Module):
    def __init__(self, c_in, c_out, num_layer, c_h):
        super().__init__()

        self.inorm         = InstanceNorm1D()
        self.in_conv       = BasicConv(c_in, c_h, kernel_size=3, stride=1, padding=1, relu=True, norm=True, norm_type='bn')
        self.conv_blocks   = nn.ModuleList([
                                            TIVEncoderBlock(c_h, c_h)
                                            for _ in range(num_layer)
                                            ])
        self.out_conv      = BasicConv(c_h, c_out, kernel_size=3, stride=1, padding=1, norm=True, norm_type='bn') 

    def forward(self, x, mask):

        x = self.in_conv(x.squeeze(1) * mask) * mask

        skips = []
        for block in self.conv_blocks:
            x = block(x * mask) * mask
            skips.append(x)
            x = self.inorm(x)
        x    = self.out_conv(x * mask) * mask
        
        return x, skips
            
class TVEncoder(nn.Module):
    def __init__(self, c_in, c_out, c_out_g, num_layer, c_h, n_emb, commit_w):
        super().__init__()

        self.in_conv       = BasicConv(c_in, c_h, kernel_size=3, stride=1, padding=1, relu=True, norm=True, norm_type='ln')
        self.conv_blocks   = nn.ModuleList([
                                            TVEncoderBlock(c_h, c_h)
                                            for _ in range(num_layer)
                                            ])
        self.out_conv      = BasicConv(c_h, c_out, kernel_size=3, stride=1, padding=1, relu=False, norm=False) #
        self.vq            = VQEmbeddingEMA(n_emb, c_out, commit_w)
        self.proj_0        = Projection(c_out, c_out_g, kernel_size=3, p_drop=0.1)
        self.proj_1        = BasicConv(c_out_g, c_out_g, kernel_size=3, stride=1, padding=1, relu=True, norm=True, norm_type='bn')
        

    def forward(self, x, mask):
        
        # mask : b, 1, t
        
        x = self.in_conv(x.squeeze(1) * mask) * mask

        skips = []
        for block in self.conv_blocks:
            x = block(x * mask) * mask
            skips.append(x)
        z_beforeVQ = self.out_conv(x * mask) * mask  # b, c, t
        z, vq_loss = self.vq(z_beforeVQ.transpose(1,2), mask) # b, t, c
            
        z_dec  = z.detach().transpose(1,2)
        z_dec  = self.proj_0(z_dec, mask)
        z_dec  = self.proj_1(z_dec * mask) * mask

        return z_beforeVQ, z_dec, vq_loss
            
class TVAdaptor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.d_k     = channels ** 0.5
        self.w_q     = nn.Linear(channels, channels, bias=False)
        self.w_k     = nn.Linear(channels, channels, bias=False)
        self.w_v     = nn.Linear(channels, channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.linear  = nn.Linear(channels, channels, bias=False)
        self.inorm2d = InstanceNorm2D()
    
    def forward(self, x, x_mask, sty, sty_mask, time):
        
        
        sty = torch.cat([time, sty], dim=-1)  # b, c, (t+1)
        # x: b, c, h, w , sty: b, c, t        
        b, c, h, w = x.shape
        
        add_mask = torch.ones((b,1,1,1)).to(sty_mask.dtype).to(sty_mask.device)        
        sty_mask = torch.cat([add_mask, sty_mask], dim=-1)
        sty_mask = sty_mask.repeat((1, h, w, 1)) # b, h, w, t

        # q: b, h, w, c,  k & v: b, t, c
        q = self.w_q(self.inorm2d(x).permute(0,2,3,1))
        k = self.w_k(sty.transpose(1,2)).unsqueeze(1)
        v = self.w_v(sty.transpose(1,2)).unsqueeze(1)
        
        attn = torch.matmul(q / self.d_k, k.transpose(-1,-2))  # b, 1, w, t
        attn = attn.masked_fill(sty_mask == 0, -1e4)
        attn = self.softmax(attn)
        
        output = torch.matmul(attn, v)
        output = self.linear(output).permute(0,3,1,2)
        output = x + output
        output = output * x_mask
        
        return output
            
class VQEmbeddingEMA(nn.Module):
    '''
    reference from: https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py
    '''
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        
        self.commitment_cost = commitment_cost
        self.decay           = decay
        self.epsilon         = epsilon

        init_bound = 1 / n_embeddings
        embedding  = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding) # only change during forward
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward(self, x, x_mask):
        
        x_mask = x_mask.transpose(1,2)
        x      = x * x_mask
        
        M, D   = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0) # calculate the distance between each ele in embedding and x

        indices   = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training: # EMA based codebook learning
            self.ema_count  = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            n               = torch.sum(self.ema_count)
            self.ema_count  = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            dw              = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding  = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = torch.sum(((x * x_mask) - (quantized.detach() * x_mask)) ** 2) / (torch.sum(x_mask) * x.size(-1))
        loss = self.commitment_cost * e_latent_loss
        
        # residual  = x - quantized
        quantized = x + (quantized - x).detach()

        # avg_probs  = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # residual   = residual * x_mask
        quantized  = quantized * x_mask

        return quantized, loss
                        
class SelfAttentionPooling(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.W      = nn.Linear(input_dim, 1)
        self.d_k    = input_dim ** 0.5
        
    def forward(self, x, time):

        x    = torch.cat([time, x], dim=1) # b, (l+1), c
        attn = self.W(x).squeeze(-1)       # b, (l+1)
        attn = F.softmax(attn, dim=-1).unsqueeze(-1)  # b, (l+1), 1
        x    = torch.sum(x * attn, dim=1)             # b, c

        return x
        
class TIVAdaptor(nn.Module):
    def __init__(self, channels):
        super(TIVAdaptor, self).__init__()

        self.mean_sap = SelfAttentionPooling(channels)
        self.std_sap  = SelfAttentionPooling(channels)
        
        self.inorm2d = InstanceNorm2D()
    
    def forward(self, x, ref, ref_lengths, time):
        
        ref_mean, ref_std = ref
        
        ref_mean = self.mean_sap(ref_mean, time).unsqueeze(-1) # B, L, C -> B, C, 1
        ref_std  = self.std_sap(ref_std, time).unsqueeze(-1)  # B, L, C -> B, C, 1 
                
        x = self.inorm2d(x) * ref_std.unsqueeze(-1) + ref_mean.unsqueeze(-1)
        
        return x
    