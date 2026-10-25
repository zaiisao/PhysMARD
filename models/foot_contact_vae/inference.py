"""Load the trained HumanML3D VAE and infer foot-contact probabilities from joints."""

import json
from pathlib import Path

import numpy as np
import torch

from .source.geometry import FOOT_JOINTS, SIDE_NAMES
from .source.model import GEOMETRY_DIM, STATE_ORDER, StructuredFootVAE, chain_marginals

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "checkpoints" / "humanml3d_best.pt"
DEFAULT_CONFIG = PACKAGE_ROOT / "training" / "config.json"
STATE_NAMES = tuple(state.name.lower() for state in STATE_ORDER)


def make_batch(joints, device):
    """Convert [T,22,3] world-space HumanML3D joints to the two-foot VAE batch."""
    joints_tensor = torch.as_tensor(joints, dtype=torch.float32, device=device)
    frames = joints_tensor.shape[0]

    left_ankle_index, left_toe_index = FOOT_JOINTS["left"]
    right_ankle_index, right_toe_index = FOOT_JOINTS["right"]

    left_ankle = joints_tensor[:, left_ankle_index]
    right_ankle = joints_tensor[:, right_ankle_index]
    left_toe = joints_tensor[:, left_toe_index]
    right_toe = joints_tensor[:, right_toe_index]

    ankles = torch.stack((left_ankle, right_ankle))
    toes = torch.stack((left_toe, right_toe))
    lengths = torch.full((len(SIDE_NAMES),), frames, dtype=torch.long, device=device)
    sides = torch.arange(len(SIDE_NAMES), dtype=torch.long, device=device)

    return {
        "ankle": ankles,
        "toe": toes,
        "lengths": lengths,
        "sides": sides,
    }


def _repeat_for_samples(value, sample_count):
    """Stack each batch item once per sample, keeping sample-major order."""
    expanded = value.unsqueeze(0).expand(sample_count, *value.shape)
    batch_size = value.shape[0]
    return expanded.reshape(sample_count * batch_size, *value.shape[1:])


def load_model(device, checkpoint=DEFAULT_CHECKPOINT):
    """Load the packaged checkpoint onto an explicitly selected device."""
    device = torch.device(device)

    # Load weights on CPU, then move the reconstructed model to the requested device.
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config_values = json.loads(DEFAULT_CONFIG.read_text())

    input_mean = checkpoint_state["encoder_input_mean"]
    input_sd = checkpoint_state["encoder_input_sd"]
    floor_height = config_values["floor_m"]
    hidden_size = config_values["model"]["hidden"]
    initial_observations = config_values["initial_observation_parameters"]

    model = StructuredFootVAE(
        input_mean,
        input_sd,
        floor_height=floor_height,
        init_air_mean=initial_observations["air_mean_m"],
        init_pitch_mean=initial_observations["pitch_mean_rad"],
        hidden=hidden_size,
    )
    model.load_state_dict(checkpoint_state, strict=True)
    model.to(device)
    model.requires_grad_(False)

    return model


@torch.no_grad()
def predict_contacts(joints, model, geometry_samples=32, seed=95095):
    """Infer seven-state contact probabilities for a [T,22,3] sequence.

    The input motion is never modified. Output side order is [left, right].
    """
    device = next(model.parameters()).device
    foot_batch = make_batch(joints, device)

    geometry_parameters = model.encode(
        foot_batch["ankle"], foot_batch["toe"], foot_batch["lengths"]
    )
    foot_count = foot_batch["ankle"].shape[0]

    # Fixed noise makes geometry sampling repeatable for a given seed.
    sample_generator = torch.Generator(device=device)
    sample_generator.manual_seed(int(seed))
    geometry_noise = torch.randn(
        (geometry_samples, foot_count, GEOMETRY_DIM),
        generator=sample_generator,
        device=device,
        dtype=foot_batch["ankle"].dtype,
    )

    sides = foot_batch["sides"]
    geometry_prior_mean = model.geometry_mean[sides]
    geometry_prior_covariance = model.geometry_cov[sides]
    repeated_geometry_parameters = geometry_parameters.repeat(geometry_samples, 1)
    repeated_prior_mean = geometry_prior_mean.repeat(geometry_samples, 1)
    repeated_prior_covariance = geometry_prior_covariance.repeat(geometry_samples, 1, 1)
    flattened_geometry_noise = geometry_noise.reshape(-1, GEOMETRY_DIM)

    # Reuse the model's posterior sampler so inference matches its training math.
    sampled_geometry, _ = model.geometry_posterior.sample_and_kl(
        repeated_geometry_parameters,
        repeated_prior_mean,
        repeated_prior_covariance,
        sample=True,
        eps=flattened_geometry_noise,
    )

    ankles_by_sample = _repeat_for_samples(foot_batch["ankle"], geometry_samples)
    toes_by_sample = _repeat_for_samples(foot_batch["toe"], geometry_samples)
    lengths_by_sample = _repeat_for_samples(foot_batch["lengths"], geometry_samples)
    frames = foot_batch["ankle"].shape[1]

    initial_state_scores, transition_state_scores = model.state_prior_log_potentials(
        lengths_by_sample, frames
    )
    frame_emission_scores, transition_emission_scores = model.foot_contact_emission(
        ankles_by_sample,
        toes_by_sample,
        sampled_geometry.reshape(-1, GEOMETRY_DIM),
        lengths_by_sample,
    )
    transition_scores = transition_state_scores + transition_emission_scores
    sampled_state_probabilities = chain_marginals(
        initial_state_scores,
        frame_emission_scores,
        transition_scores,
        lengths_by_sample,
    )
    sampled_state_probabilities = sampled_state_probabilities.reshape(
        geometry_samples, foot_count, frames, len(STATE_NAMES)
    )
    # Marginalize foot geometry by averaging state probabilities across samples.
    state_probabilities = sampled_state_probabilities.mean(0)

    return {
        "state_probability": state_probabilities.cpu().numpy(),
        "state_names": np.asarray(STATE_NAMES),
        "side_names": np.asarray(SIDE_NAMES),
    }
