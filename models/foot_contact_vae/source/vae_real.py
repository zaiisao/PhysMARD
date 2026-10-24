"""Real-data VAE variants and shared training utilities (GPU).

RealVAE = StructuredFootVAE with a configurable encoder:
    hidden, layers      width / number of dilated conv layers (kernel 5, dilation 1, 2, 4, ...)
    pooling             'mean' (original) or 'meanmax' (masked mean and max over time)
    standardize         standardize the 5 per-frame inputs with training-set mean / sd
Training protocol is identical across variants (constant-lr Adam, 8 geometry samples, minibatch 16 in length buckets).
Model selection uses validation ELBO on clips disjoint from train and test.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import StructuredFootVAE
from real_heldout import (clip_ids, sequences, collate, batches, mean_elbo, amortization_gap)
from real_train import FLOOR

DEV = torch.device('cuda')
OUT = Path(__file__).resolve().parent / 'runs_real'
SCALES = ('log_contact_sd', 'air_mean', 'log_air_sd', 'log_flat_sd', 'pitch_mean', 'log_pitch_sd',
          'log_stick_sd', 'log_slide_sd', 'log_free_sd')
FIXED_DECLARED = ('log_slide_sd',)   # sliding is rare in real motion, so its scale is not identifiable: stays at the declared prior


class RealVAE(StructuredFootVAE):
    def __init__(self, hidden=64, layers=2, pooling='mean', standardize=False, stats=None):
        super().__init__(context_dim=4, fixed_floor_height=FLOOR, hidden=hidden)
        self.pooling, self.standardize = pooling, standardize
        mods, c_in = [], 9
        for i in range(layers):
            mods += [nn.Conv1d(c_in, hidden, 5, padding=2 * 2 ** i, dilation=2 ** i), nn.GELU()]
            c_in = hidden
        self.encoder = nn.Sequential(*mods)
        self.cont_head = nn.Linear(hidden * (2 if pooling == 'meanmax' else 1), 27)
        nn.init.zeros_(self.cont_head.weight); nn.init.zeros_(self.cont_head.bias)
        mean, sd = stats if stats is not None else (torch.zeros(9), torch.ones(9))
        self.register_buffer('input_mean', mean.float()); self.register_buffer('input_sd', sd.float())

    def encode(self, u, ankle, toe, lengths):
        x = self.encoder_inputs(u, ankle, toe)
        if self.standardize:
            x = (x - self.input_mean) / self.input_sd
        valid = (torch.arange(u.shape[1], device=u.device)[None] < lengths[:, None])[..., None]
        feats = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        pooled = (feats * valid).sum(1) / lengths[:, None]
        if self.pooling == 'meanmax':
            pooled = torch.cat([pooled, feats.masked_fill(~valid, -1e4).amax(1)], -1)
        return self.cont_head(pooled)


def input_stats(rows):
    """Mean / sd of the 5 per-frame encoder inputs over training frames (context u columns left unscaled)."""
    probe = StructuredFootVAE(context_dim=4, fixed_floor_height=FLOOR).to(DEV)
    d = collate(rows[:400])
    x = probe.encoder_inputs(d['context'], d['ankle'], d['toe'])
    valid = torch.arange(x.shape[1], device=DEV)[None] < d['lengths'][:, None]
    x = x[valid]
    mean, sd = x.mean(0), x.std(0)
    mean[5:], sd[5:] = 0.0, 1.0
    return mean.cpu(), sd.cpu()


def data(n_train, n_val=30, n_test=50):
    train_ids = clip_ids('train', n_train, 1)
    val_ids = clip_ids('train', n_val, 2, exclude=train_ids)
    test_ids = clip_ids('test', n_test, 3)
    return sequences(train_ids), sequences(val_ids), sequences(test_ids)


def train(model, train_rows, val_rows, epochs, free, log, lr=1e-3, seed=731, init_best=-float('inf')):
    """Train parameters whose names are not in `frozen`; returns the best-validation-ELBO state and the history."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    frozen = tuple(n for n in SCALES if n not in free)
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith(frozen))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    val_batches = [collate(b) for b in batches(val_rows, 32, False, rng)]
    best, best_state, history, t0 = init_best, None, [], time.time()
    for ep in range(epochs):
        model.train(); losses = []
        for b in batches(train_rows, 16, True, rng):
            d = collate(b)
            loss = -mean_elbo(model, d, 8).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
            losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            v = float(np.mean([float(mean_elbo(model, d, 8).mean()) for d in val_batches]))
        history.append(dict(epoch=ep + 1, train_elbo=-float(np.mean(losses)), val_elbo=v))
        if v > best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
        if ep % 20 == 0 or ep == epochs - 1:
            log(json.dumps(history[-1]) + f' t={time.time() - t0:.0f}s')
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, history


def held_out_summary(model, test_rows, in_rows):
    model.eval()
    held, _ = amortization_gap(model, test_rows, 'held-out')
    seen, _ = amortization_gap(model, in_rows, 'train (in-sample)')
    return held, seen
