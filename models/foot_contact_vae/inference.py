"""Load the trained HumanML3D VAE and infer foot-contact probabilities from joints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .source.model import GEOMETRY_DIM, StructuredFootVAE
from .source.model import chain_log_partition

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "checkpoints" / "humanml3d_best.pt"
DEFAULT_CONFIG = PACKAGE_ROOT / "training" / "config.json"
STATE_NAMES = (
    "airborne", "flat_sticking", "flat_sliding", "forefoot_sticking",
    "forefoot_sliding", "toe_tip_sticking", "toe_tip_sliding",
)


def geometry_posterior(model, batch):
    """Return q(g)'s mean, Cholesky factor, covariance, and analytic KL per foot."""
    raw = model.encode(batch["ankle"], batch["toe"], batch["lengths"])
    eta_mean, tri_raw = raw[:, :GEOMETRY_DIM], raw[:, GEOMETRY_DIM:]
    tri = torch.tril_indices(GEOMETRY_DIM, GEOMETRY_DIM, device=raw.device)
    diagonal = tri[0] == tri[1]
    eta_chol = raw.new_zeros((raw.shape[0], GEOMETRY_DIM, GEOMETRY_DIM))
    eta_chol[:, tri[0], tri[1]] = torch.where(diagonal[None], tri_raw.exp(), tri_raw)
    prior_mean = model.geometry_mean[batch["sides"]]
    prior_chol = torch.linalg.cholesky(model.geometry_cov[batch["sides"]])
    q_mean = prior_mean + torch.einsum("bij,bj->bi", prior_chol, eta_mean)
    q_chol = prior_chol @ eta_chol
    q_cov = q_chol @ q_chol.transpose(-1, -2)
    eta_cov = eta_chol @ eta_chol.transpose(-1, -2)
    kl = 0.5 * (
        eta_mean.square().sum(-1)
        + torch.diagonal(eta_cov, dim1=-2, dim2=-1).sum(-1)
        - GEOMETRY_DIM
        - 2 * torch.log(torch.diagonal(eta_chol, dim1=-2, dim2=-1)).sum(-1)
    )
    return q_mean, q_chol, q_cov, kl


def make_batch(joints, device):
    """Convert [T,22,3] world-space HumanML3D joints to the two-foot VAE batch."""
    tensor = torch.as_tensor(joints, dtype=torch.float32, device=device)
    frames = tensor.shape[0]
    return {
        "ankle": torch.stack((tensor[:, 7], tensor[:, 8])),
        "toe": torch.stack((tensor[:, 10], tensor[:, 11])),
        "lengths": torch.full((2,), frames, dtype=torch.long, device=device),
        "sides": torch.tensor((0, 1), dtype=torch.long, device=device),
    }


def _repeat(value, samples):
    return value.unsqueeze(0).expand(samples, *value.shape).reshape(samples * value.shape[0], *value.shape[1:])


def elbo_mc(model, batch, geometry_samples, eps=None):
    """Monte Carlo ELBO with analytic KL for a batch of foot sequences."""
    q_mean, q_chol, _, kl = geometry_posterior(model, batch)
    if eps is None:
        eps = torch.randn((geometry_samples, *q_mean.shape), device=q_mean.device, dtype=q_mean.dtype)
    geometry = q_mean[None] + torch.einsum("bij,sbj->sbi", q_chol, eps)
    repeated = {
        key: _repeat(batch[key], geometry_samples)
        for key in ("ankle", "toe", "lengths", "sides")
    }
    p_init, p_edge = model.state_prior_log_potentials(repeated["lengths"], repeated["ankle"].shape[1])
    frame, emission_edge = model.foot_contact_emission(
        repeated["ankle"], repeated["toe"], geometry.reshape(-1, GEOMETRY_DIM), repeated["lengths"]
    )
    log_z = chain_log_partition(p_init, frame, p_edge + emission_edge, repeated["lengths"])
    log_z = log_z.reshape(geometry_samples, q_mean.shape[0])
    elbo = log_z.mean(0) - kl
    return elbo, {"log_z_samples": log_z, "kl_geometry": kl, "q_geometry_mean": q_mean,
                  "q_geometry_cholesky": q_chol}


def _forward_backward(model, batch, geometry):
    """Exact HMM marginals for each sampled foot geometry."""
    batch_size, frames = batch["ankle"].shape[:2]
    lengths = batch["lengths"]
    p_init, p_edge = model.state_prior_log_potentials(lengths, frames)
    frame, emission_edge = model.foot_contact_emission(batch["ankle"], batch["toe"], geometry, lengths)
    edge = p_edge + emission_edge

    alpha = [p_init + frame[:, 0]]
    for t in range(1, frames):
        alpha.append(torch.logsumexp(alpha[-1][:, :, None] + edge[:, t - 1], dim=1) + frame[:, t])
    alpha = torch.stack(alpha, dim=1)

    beta = [None] * frames
    beta[-1] = torch.zeros_like(alpha[:, -1])
    for t in range(frames - 2, -1, -1):
        beta[t] = torch.logsumexp(
            edge[:, t] + frame[:, t + 1, None, :] + beta[t + 1][:, None, :], dim=-1
        )
    beta = torch.stack(beta, dim=1)
    gamma = torch.softmax(alpha + beta, dim=-1)

    return gamma


def load_model(checkpoint=DEFAULT_CHECKPOINT, device="auto"):
    """Load the packaged checkpoint; device='auto' selects CUDA when available."""
    checkpoint = Path(checkpoint)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = json.loads(DEFAULT_CONFIG.read_text())
    model = StructuredFootVAE(
        state["encoder_input_mean"], state["encoder_input_sd"],
        floor_height=config["floor_m"], hidden=config["model"]["hidden"],
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def predict_contacts(joints, model, geometry_samples=32, seed=95095):
    """Infer seven-state contact probabilities for a [T,22,3] sequence.

    The input motion is never modified. Context is zero, as in the trained
    HumanML3D experiment. Output arrays use side order [left, right].
    """
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3) or joints.shape[0] < 2:
        raise ValueError(f"Expected joints with shape [T,22,3] and T >= 2, got {joints.shape}")
    if not np.isfinite(joints).all():
        raise ValueError("Input joints contain NaN or infinite values")
    if geometry_samples < 1:
        raise ValueError("geometry_samples must be positive")

    device = next(model.parameters()).device
    batch = make_batch(joints, device)
    q_mean, q_chol, _, _ = geometry_posterior(model, batch)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    eps = torch.randn((geometry_samples, 2, GEOMETRY_DIM), generator=generator,
                      device=device, dtype=batch["ankle"].dtype)
    geometry = q_mean[None] + torch.einsum("bij,sbj->sbi", q_chol, eps)

    repeated = {
        key: _repeat(batch[key], geometry_samples)
        for key in ("ankle", "toe", "lengths", "sides")
    }
    gamma_samples = _forward_backward(model, repeated, geometry.reshape(-1, GEOMETRY_DIM))
    frames = len(joints)
    gamma_samples = gamma_samples.reshape(geometry_samples, 2, frames, 7)
    gamma = gamma_samples.mean(0)

    return {
        "state_probability": gamma.cpu().numpy(),
        "state_names": np.asarray(STATE_NAMES),
        "side_names": np.asarray(("left", "right")),
    }
