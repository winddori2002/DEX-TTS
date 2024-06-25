# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import numpy as np
#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss(torch.nn.Module):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, n_feats=80, loss_type='base'):
        super().__init__()
        self.P_mean     = P_mean
        self.P_std      = P_std
        self.sigma_data = sigma_data
        self.n_feats    = n_feats
        self.loss_type  = loss_type

    def forward(self, precond_model, x0, mask, mu, spk=None, mask_ratio=0):

        rnd_normal = torch.randn([x0.shape[0], 1,  1], device=x0.device)
        sigma      = (rnd_normal * self.P_std + self.P_mean).exp()
        snr        = 1 / sigma**2

        if self.loss_type == 'base':
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            
        if self.loss_type.startswith('base_min_'):
            k      = float(self.loss_type.split('base_min_')[-1])
            snr    = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            weight = torch.stack([snr, k * torch.ones_like(sigma)], dim=1).min(dim=1)[0]
            
        if self.loss_type.startswith('base_log_'):
            k      = float(self.loss_type.split('base_log_')[-1])
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            weight[torch.where(weight >= k)] = torch.log(weight[torch.where(weight >= k)]) + (k - np.log(k))

        elif self.loss_type.startswith('min_snr_'):
            k      = float(self.loss_type.split('min_snr_')[-1])
            weight = torch.stack([snr, k * torch.ones_like(sigma)], dim=1).min(dim=1)[0]
            
        elif self.loss_type.startswith('max_snr_'):
            k      = float(self.loss_type.split('max_snr_')[-1])
            weight = torch.stack([snr, k * torch.ones_like(sigma)], dim=1).max(dim=1)[0]

        elif self.loss_type == 'snr':
            weight = snr

        elif self.loss_type == 'inv_snr':
            weight = 1. / snr
        
        n    = (torch.randn_like(x0) + mu) * sigma
        D_yn = precond_model(x0 + n, sigma, mask, mu, spk=spk, mask_ratio=mask_ratio)
        loss = torch.sum(weight * ((D_yn - x0) ** 2)) / torch.sum(mask * self.n_feats)

        return loss

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

class EDMPrecond(torch.nn.Module):
    def __init__(self,
        model, 
        sigma_min  = 0,                # Minimum supported noise level.
        sigma_max  = float('inf'),     # Maximum supported noise level.
        sigma_data = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()

        self.model      = model
        self.sigma_min  = sigma_min
        self.sigma_max  = sigma_max
        self.sigma_data = sigma_data

    def forward(self, x, sigma, mask, mu, spk=None, mask_ratio=0):

        sigma   = sigma.reshape(-1, 1, 1)
        c_skip  = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out   = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in    = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x), mask, mu, c_noise.flatten(), spk=spk, mask_ratio=mask_ratio)
        D_x = c_skip * x + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

# Proposed EDM sampler (Algorithm 2).

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(net, latents, mask=None, mu=None, spk=None, class_labels=None, randn_like=torch.randn_like, num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none', epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,  S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma       = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv   = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma       = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv   = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def    = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def    = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d   = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps  = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps  = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u         = torch.zeros(M + 1, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered  = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma       = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv   = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma       = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv  = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma       = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv   = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s       = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s       = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h        = t_next - t_hat
        # denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        denoised = net(x_hat / s(t_hat), sigma(t_hat), mask, mu, spk=spk)
        d_cur    = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime  = x_hat + alpha * h * d_cur
        t_prime  = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            # denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            denoised = net(x_prime / s(t_prime), sigma(t_prime), mask, mu, spk=spk)
            d_prime  = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next   = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next