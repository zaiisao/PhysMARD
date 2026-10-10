"""PIDM sample-estimation helpers, copied verbatim from the reference implementation.

Source: https://github.com/jhbastek/PhysicsInformedDiffusionModels
        commit 733aeb0fe8e56f8d1b3aaf9ff24f340438c20d9a, src/denoising_toy_utils.py
License: MIT, Copyright (c) 2024 Jan-Hendrik Bastek.

The function bodies below are unmodified. ddim_sample_x0 with reduced_n_steps=0
maps x_t -> x_1 -> x_0 (zero-based indices t -> 0 -> -1) as in the manuscript.
"""

import numpy as np
import torch


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def ddim_sample_x0(xt, t, model, shape, reduced_n_steps, ddim_sampling_eta, diff_dict, model_pred_mode = 'eps'):

    batch, device, sampling_timesteps, eta = shape[0], diff_dict['alphas'].device, reduced_n_steps, ddim_sampling_eta

    if len(t) == 1:
        batch_t = torch.ones(batch, device=device, dtype=torch.long)*t
    else:
        batch_t = t

    batch_t = batch_t.cpu().numpy()
    seqs = []
    seqs_next = []
    for t_idx, t in enumerate(batch_t):
        seq = list(np.linspace(0, batch_t[t_idx], sampling_timesteps+2, endpoint=True, dtype=float)) # evenly spread from 0 to current t
        seq = list(map(int, seq))
        seqs.append(list(reversed(seq)))
        seq_next = [-1] + list(seq[:-1])
        seqs_next.append(list(reversed(seq_next)))
        seq = None

    # tranpose to have time as first dimension
    cur_times = torch.tensor(seqs, device=device).T
    next_times = torch.tensor(seqs_next, device=device).T

    time_pairs = list(zip(cur_times, next_times)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    cur_x = xt
    x0_pred = None

    for t, t_next in time_pairs:

        # create mask for those timesteps that are equal
        mask = (t == t_next).float().unsqueeze(-1)

        if model_pred_mode == 'eps':
            eps_theta = model(cur_x, t)
            x0_pred = predict_start_from_noise(cur_x, t, eps_theta, diff_dict)
        elif model_pred_mode == 'x0':
            model_pred = model(cur_x, t)
            x0_pred = model_pred
            mean = (
                extract(diff_dict['posterior_mean_coef1'], t, cur_x) * x0_pred +
                extract(diff_dict['posterior_mean_coef2'], t, cur_x) * cur_x
            )
            eps_theta = predict_noise_from_mean(cur_x, t, mean, diff_dict)
        elif model_pred_mode == 'mu':
            model_pred = model(cur_x, t)
            eps_theta = predict_noise_from_mean(cur_x, t, model_pred, diff_dict)
            x0_pred = predict_start_from_noise(cur_x, t, eps_theta, diff_dict)
        else:
            raise ValueError('model_pred_mode not recognized.')

        if t_next[0] < 0: # this happens when we predict x0, should never happen during training
            # assert that all next timesteps are equal
            assert torch.all(t_next == -1), 'Next timesteps should be -1, otherwise this is inconsistent.'
            cur_x = x0_pred
            continue

        alpha = extract(diff_dict['alphas_prod'], t, cur_x)
        alpha_next = extract(diff_dict['alphas_prod'], t_next, cur_x)

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(cur_x)

        # only change cur_x where t != t_next
        cur_x_ = x0_pred * alpha_next.sqrt() + \
                c * eps_theta + \
                sigma * noise

        cur_x = mask * cur_x + (1 - mask) * cur_x_

    return cur_x

def predict_start_from_noise(x_t, t, noise, diff_dict):
    return (
        extract(diff_dict['sqrt_recip_alphas_cumprod'], t, x_t) * x_t -
        extract(diff_dict['sqrt_recipm1_alphas_cumprod'], t, x_t) * noise
    )

def predict_noise_from_mean(x_t, t, mean_t, diff_dict):
    return (
        extract(diff_dict['sqrt_recip_alphas'], t, mean_t) * x_t - mean_t
    ) / extract(diff_dict['noise_mean_coeff'], t, mean_t)
