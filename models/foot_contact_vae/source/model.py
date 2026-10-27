from enum import IntEnum

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from .geometry import load_support_geometry

FPS = 20.0
ENCODER_FEATURES = (
    "ankle_height", "toe_height", "ankle_to_toe_angle", "ankle_speed", "toe_speed",
)
NUM_ENCODER_FEATURES = len(ENCODER_FEATURES)
GEOMETRY_DIM = 6


class SupportPoint(IntEnum):
    HEEL = 0
    PAD = 1
    TOE_TIP = 2


class StepModel(IntEnum):
    FREE = 0
    STICK = 1
    SLIDE = 2


class FootState(IntEnum):
    AIRBORNE = 0
    FLAT_STICKING = 1
    FLAT_SLIDING = 2
    FOREFOOT_STICKING = 3
    FOREFOOT_SLIDING = 4
    TOE_TIP_STICKING = 5
    TOE_TIP_SLIDING = 6


NUM_STATES = len(FootState)

SUPPORT_POINT_BY_STATE = {
    FootState.AIRBORNE: SupportPoint.PAD,
    FootState.FLAT_STICKING: SupportPoint.PAD,
    FootState.FLAT_SLIDING: SupportPoint.PAD,
    FootState.FOREFOOT_STICKING: SupportPoint.PAD,
    FootState.FOREFOOT_SLIDING: SupportPoint.PAD,
    FootState.TOE_TIP_STICKING: SupportPoint.TOE_TIP,
    FootState.TOE_TIP_SLIDING: SupportPoint.TOE_TIP,
}

FLAT_STATES = (FootState.FLAT_STICKING, FootState.FLAT_SLIDING)
FOREFOOT_STATES = (FootState.FOREFOOT_STICKING, FootState.FOREFOOT_SLIDING)
TOE_TIP_STATES = (FootState.TOE_TIP_STICKING, FootState.TOE_TIP_SLIDING)

STICKING_SLIDING_GROUPS = (FLAT_STATES, FOREFOOT_STATES, TOE_TIP_STATES)

STATE_ORDER = tuple(FootState)


def step_model_codes():
    """Return each state's motion model for every previous-to-current transition."""
    codes = [[StepModel.FREE for _ in STATE_ORDER] for _ in STATE_ORDER]
    for sticking, sliding in STICKING_SLIDING_GROUPS:
        codes[sticking][sticking] = StepModel.STICK
        codes[sticking][sliding] = StepModel.SLIDE
        codes[sliding][sticking] = StepModel.SLIDE
        codes[sliding][sliding] = StepModel.SLIDE
    return codes


def foot_frame(ankle, toe):
    """Return ankle-to-toe direction, perpendicular-up direction, length and angle per frame."""
    d = toe - ankle
    length = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
    ankle_to_toe_direction = d / length
    world_up = torch.zeros_like(ankle_to_toe_direction)
    world_up[..., 1] = 1.0
    world_up_component_along_foot = torch.sum(
        world_up * ankle_to_toe_direction, dim=-1, keepdim=True
    )
    world_up_perpendicular_to_foot = (
        world_up - world_up_component_along_foot * ankle_to_toe_direction
    )
    upward_perpendicular_length = torch.linalg.vector_norm(
        world_up_perpendicular_to_foot, dim=-1, keepdim=True
    )
    upward_perpendicular_direction = world_up_perpendicular_to_foot / upward_perpendicular_length
    ankle_to_toe_angle = torch.asin(ankle_to_toe_direction[..., 1:2])
    return ankle_to_toe_direction, upward_perpendicular_direction, length, ankle_to_toe_angle


def support_point_positions(ankle, toe, geom):
    """World positions [B,T,3,3] of heel, pad and tip.

    Each geometry point has offsets (ankle-to-toe, upward-perpendicular), measured in foot lengths.
    """
    ankle_to_toe_direction, upward_perpendicular_direction, length, _ = foot_frame(ankle, toe)
    coeff = geom.view(-1, 1, 3, 2)
    offset = (
        coeff[..., 0:1] * ankle_to_toe_direction[:, :, None]
        + coeff[..., 1:2] * upward_perpendicular_direction[:, :, None]
    )
    return ankle[:, :, None] + length[:, :, None] * offset


def flat_ankle_to_toe_angle(geom):
    """Ankle-to-toe angle at which the heel and pad are level.

    A point (a, b) sits at height L (a sin(angle) + b cos(angle)) above the ankle, so heel and pad are level when
    tan(angle) = -(b_heel - b_pad) / (a_heel - a_pad).
    """
    g = geom.view(-1, 3, 2)

    heel_normal, pad_normal = g[:, SupportPoint.HEEL, 1], g[:, SupportPoint.PAD, 1]
    heel_along, pad_along = g[:, SupportPoint.HEEL, 0], g[:, SupportPoint.PAD, 0]

    normal_difference = heel_normal - pad_normal
    along_difference = heel_along - pad_along

    return torch.atan(-normal_difference / along_difference)


def log_matmul(a, b):
    """Matrix product in the log semiring: C[..., i, k] = logsumexp_j (A[..., i, j] + B[..., j, k])."""
    return torch.logsumexp(a[..., :, :, None] + b[..., None, :, :], dim=-2)


def log_prefix_scan(m):
    """Inclusive scan over dim 1 of log-semiring matrices: out[:, t] = m[:, 0] (x) ... (x) m[:, t] (Hillis-Steele)."""
    d = 1
    while d < m.shape[1]:
        m = torch.cat([m[:, :d], log_matmul(m[:, :-d], m[:, d:])], dim=1)
        d *= 2
    return m


def chain_log_partition(init, frame, edge, lengths):
    """Compute the exact state-marginalized log score with a parallel prefix scan."""
    t, k = frame.shape[1:]
    eye = torch.full((k, k), -1e9, dtype=frame.dtype, device=frame.device)
    eye.fill_diagonal_(0.0)
    step_valid = (torch.arange(1, t, device=frame.device)[None] < lengths[:, None])[..., None, None]
    transitions = torch.where(step_valid, edge + frame[:, 1:, None, :], eye)

    alpha0 = init + frame[:, 0]
    prefix = log_prefix_scan(transitions)
    final_alpha = torch.logsumexp(
        alpha0[:, :, None] + prefix[:, -1], dim=-2
    )
    return torch.logsumexp(final_alpha, dim=-1)


def chain_marginals(init, frame, edge, lengths):
    """Return exact per-frame state probabilities for a batch of HMM chains."""
    frames = frame.shape[1]
    alpha = init + frame[:, 0]
    forward = [alpha]
    for t in range(1, frames):
        next_alpha = torch.logsumexp(alpha[:, :, None] + edge[:, t - 1], dim=1) + frame[:, t]
        alpha = torch.where((lengths > t)[:, None], next_alpha, alpha)
        forward.append(alpha)

    beta = torch.zeros_like(forward[-1])
    backward = [beta]
    for t in range(frames - 2, -1, -1):
        previous_beta = torch.logsumexp(
            edge[:, t] + frame[:, t + 1, None, :] + beta[:, None, :], dim=-1
        )
        beta = torch.where((lengths > t + 1)[:, None], previous_beta, torch.zeros_like(beta))
        backward.append(beta)

    backward.reverse()
    return torch.softmax(torch.stack(forward, dim=1) + torch.stack(backward, dim=1), dim=-1)

class FootContactEmission(nn.Module):
    """Gaussian observation factors for support height, angle, and horizontal motion."""

    def __init__(self, init_air_mean=0.1, init_angle_mean=-0.3):
        super().__init__()

        self.air_mean = nn.Parameter(torch.tensor(init_air_mean))
        self.pitch_mean = nn.Parameter(torch.tensor(init_angle_mean))

        self.raw_contact_sd = nn.Parameter(torch.zeros(()))
        self.raw_air_sd = nn.Parameter(torch.zeros(()))
        self.raw_flat_pitch_sd = nn.Parameter(torch.zeros(()))
        self.raw_pitch_sd = nn.Parameter(torch.zeros(()))
        self.raw_stick_speed_sd = nn.Parameter(torch.zeros(()))
        self.raw_slide_speed_sd = nn.Parameter(torch.zeros(()))
        self.raw_free_speed_sd = nn.Parameter(torch.zeros(()))

        support_point = torch.tensor(list(SUPPORT_POINT_BY_STATE.values()))
        step_codes = torch.tensor(step_model_codes())
        is_contact = torch.tensor([state != FootState.AIRBORNE for state in STATE_ORDER])
        is_flat = torch.tensor([state in FLAT_STATES for state in STATE_ORDER])

        self.register_buffer("support_point", support_point, persistent=False)
        self.register_buffer("step_codes", step_codes, persistent=False)
        self.register_buffer("is_contact", is_contact, persistent=False)
        self.register_buffer("is_flat", is_flat, persistent=False)

    def forward(self, ankle, toe, geom, lengths):
        frames = ankle.shape[1]
        points = support_point_positions(ankle, toe, geom)
        _, _, _, ankle_to_toe_angle = foot_frame(ankle, toe)

        support_height = points[..., 1][..., self.support_point]

        contact_height_loc = 0.0
        contact_height_scale = F.softplus(self.raw_contact_sd)

        airborne_height_loc = self.air_mean
        airborne_height_scale = F.softplus(self.raw_air_sd)

        contact_height = Normal(contact_height_loc, contact_height_scale)
        airborne_height = Normal(airborne_height_loc, airborne_height_scale)

        height_score = torch.where(
            self.is_contact,
            contact_height.log_prob(support_height),
            airborne_height.log_prob(support_height),
        )

        flat_angle_loc = flat_ankle_to_toe_angle(geom)[:, None, None]
        flat_angle_scale = F.softplus(self.raw_flat_pitch_sd)

        other_angle_loc = self.pitch_mean
        other_angle_scale = F.softplus(self.raw_pitch_sd)

        flat_angle_dist = Normal(flat_angle_loc, flat_angle_scale)
        other_angle_dist = Normal(other_angle_loc, other_angle_scale)

        angle_score = torch.where(
            self.is_flat,
            flat_angle_dist.log_prob(ankle_to_toe_angle),
            other_angle_dist.log_prob(ankle_to_toe_angle),
        )
        frame_score = height_score + angle_score

        support_displacement = points[:, 1:] - points[:, :-1]
        x_displacement = support_displacement[..., 0]
        z_displacement = support_displacement[..., 2]

        horizontal_displacement = torch.stack((x_displacement, z_displacement), dim=-1)
        horizontal_velocity = FPS * horizontal_displacement

        speed_scales = (
            F.softplus(self.raw_free_speed_sd),
            F.softplus(self.raw_stick_speed_sd),
            F.softplus(self.raw_slide_speed_sd),
        )

        step_scores = torch.stack([
            Normal(0.0, scale).log_prob(horizontal_velocity).sum(-1)
            for scale in speed_scales
        ], dim=-1)

        destination_points = self.support_point[None, :].expand(NUM_STATES, -1)
        transition_score = step_scores[:, :, destination_points, self.step_codes]

        valid_frames = (
            torch.arange(frames, device=ankle.device)[None] < lengths[:, None]
        )[..., None]
        valid_steps = (
            torch.arange(frames - 1, device=ankle.device)[None] < (lengths - 1)[:, None]
        )[..., None, None]

        return frame_score * valid_frames, transition_score * valid_steps

class GeometryPosterior(nn.Module):
    """Encoder and full-covariance q(g|x,side), parameterized relative to the geometry prior."""

    def __init__(self, hidden):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(NUM_ENCODER_FEATURES, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
        )

        # 6 + (6 * (6 + 1) // 2) = 6 (means) + 21 (lower-triangular scale parameters) = 27.
        # Division by 2 is to prevent counting the pairs twice.
        scale_matrix_parameter_count = GEOMETRY_DIM * (GEOMETRY_DIM + 1) // 2
        posterior_output_size = GEOMETRY_DIM + scale_matrix_parameter_count
        self.cont_head = nn.Linear(hidden, posterior_output_size)

        nn.init.zeros_(self.cont_head.weight)
        nn.init.zeros_(self.cont_head.bias)

    def forward(self, inputs, lengths):
        frame_ids = torch.arange(inputs.shape[1], device=inputs.device)
        valid = frame_ids.unsqueeze(0) < lengths.unsqueeze(1)

        features = self.encoder(inputs.transpose(1, 2)).transpose(1, 2)
        pooled = (features * valid[..., None]).sum(1) / lengths[:, None]

        return self.cont_head(pooled)

    def sample_and_kl(self, encoder_output, prior_mean, prior_covariance, noise):
        """Return geometry samples [K, B, 6] and one KL value per input [B]."""
        # encoder_output has 27 values in total;
        # The first six values of encoder_output are mean adjustments in coordinates
        # standardized by the geometry prior; the remaining 21 values are for the
        # lower-triangular posterior scale matrix.
        relative_mean = encoder_output[:, :GEOMETRY_DIM]
        scale_parameters = encoder_output[:, GEOMETRY_DIM:]

        row, column = torch.tril_indices(
            GEOMETRY_DIM, GEOMETRY_DIM, device=encoder_output.device
        )
        diagonal = row == column
        scale_values = torch.where(diagonal, scale_parameters.exp(), scale_parameters)

        relative_scale_tril = torch.zeros(
            len(encoder_output), GEOMETRY_DIM, GEOMETRY_DIM,
            device=encoder_output.device, dtype=encoder_output.dtype,
        )
        relative_scale_tril[:, row, column] = scale_values

        standard_prior_mean = torch.zeros_like(relative_mean)
        standard_prior_scale_tril = torch.eye(
            GEOMETRY_DIM, device=encoder_output.device, dtype=encoder_output.dtype
        )

        standardized_posterior = MultivariateNormal(relative_mean, scale_tril=relative_scale_tril)
        standard_normal_prior = MultivariateNormal(
            standard_prior_mean, scale_tril=standard_prior_scale_tril
        )

        geometry_kl = kl_divergence(standardized_posterior, standard_normal_prior)

        standardized_geometry = relative_mean + torch.matmul(
            relative_scale_tril, noise.unsqueeze(-1)
        ).squeeze(-1)

        geometry = prior_mean + torch.matmul(
            torch.linalg.cholesky(prior_covariance), standardized_geometry.unsqueeze(-1)
        ).squeeze(-1)

        return geometry, geometry_kl


class StructuredFootVAE(nn.Module):
    """HumanML3D structured foot-contact VAE with normalized geometry encoder inputs."""

    def __init__(self, input_mean, input_sd, hidden):
        super().__init__()

        if len(input_mean) != NUM_ENCODER_FEATURES or len(input_sd) != NUM_ENCODER_FEATURES:
            raise ValueError(f"Expected {NUM_ENCODER_FEATURES} encoder statistics: {', '.join(ENCODER_FEATURES)}")

        # Learned initial and transition probabilities over the seven contact states.
        self.prior_init_logits = nn.Parameter(torch.zeros(NUM_STATES))
        self.prior_transition_logits = nn.Parameter(torch.zeros(NUM_STATES, NUM_STATES))

        self.geometry_posterior = GeometryPosterior(hidden)
        self.foot_contact_emission = FootContactEmission()

        geometry = load_support_geometry()

        # Two feet; each has six coordinates (heel, pad, tip × lengthwise and upward).
        # Means have shape [2, 6], and covariances have shape [2, 6, 6].
        # Covariance captures how point coordinates vary together across body shapes.
        geometry_mean = torch.tensor([geometry["left"]["mean"], geometry["right"]["mean"]])
        geometry_covariance = torch.tensor([geometry["left"]["covariance"], geometry["right"]["covariance"]])

        input_mean = torch.as_tensor(input_mean, dtype=torch.float32)
        input_sd = torch.as_tensor(input_sd, dtype=torch.float32)

        self.register_buffer("geometry_mean", geometry_mean)
        self.register_buffer("geometry_cov", geometry_covariance)
        self.register_buffer("encoder_input_mean", input_mean)
        self.register_buffer("encoder_input_sd", input_sd)

    def state_prior_log_potentials(self, lengths, frames):
        """Return initial-state and transition log-probabilities for each sequence."""
        batch = len(lengths)
        p_init = F.log_softmax(self.prior_init_logits, -1).expand(batch, -1)
        edge = F.log_softmax(self.prior_transition_logits, -1)
        p_edge = edge.expand(batch, frames - 1, NUM_STATES, NUM_STATES)
        return p_init, p_edge

    def encoder_inputs(self, ankle, toe):
        """Five per-frame foot features: ankle/toe height, angle, and ankle/toe speed."""
        _, _, _, ankle_to_toe_angle = foot_frame(ankle, toe)

        ankle_height, toe_height = ankle[..., 1:2], toe[..., 1:2]
        ankle_velocity = FPS * torch.linalg.vector_norm(
            ankle[:, 1:] - ankle[:, :-1], dim=-1, keepdim=True
        )
        toe_velocity = FPS * torch.linalg.vector_norm(
            toe[:, 1:] - toe[:, :-1], dim=-1, keepdim=True
        )
        ankle_speed = torch.cat([ankle_velocity, ankle_velocity[:, -1:]], dim=1)
        toe_speed = torch.cat([toe_velocity, toe_velocity[:, -1:]], dim=1)

        return ankle_height, toe_height, ankle_to_toe_angle, ankle_speed, toe_speed

    def encode(self, ankle, toe, lengths):
        """Return geometry-posterior parameters pooled over valid frames."""
        features = torch.cat(self.encoder_inputs(ankle, toe), dim=-1)
        features = (features - self.encoder_input_mean) / self.encoder_input_sd
        return self.geometry_posterior(features, lengths)

    def forward(self, ankle, toe, lengths, foot_sides, samples, generator=None):
        """Estimate the ELBO, which is defined as:

        L(x) = (1/K) sum_k log p(c | x, g_k) - KL(q(g | x) || p(g)),
        where g_k ~ q(g | x).

        Contact states c are marginalized exactly.
        """
        encoder_output = self.encode(ankle, toe, lengths)

        batch_size = len(lengths)
        geometry_noise = torch.randn(
            samples, batch_size, GEOMETRY_DIM,
            device=ankle.device, dtype=ankle.dtype, generator=generator,
        )

        geometry, geometry_kl = self.geometry_posterior.sample_and_kl(
            encoder_output,
            self.geometry_mean[foot_sides],
            self.geometry_cov[foot_sides],
            geometry_noise,
        )

        ankle = ankle.repeat(samples, 1, 1)
        toe = toe.repeat(samples, 1, 1)
        geometry = geometry.reshape(-1, GEOMETRY_DIM)
        lengths = lengths.repeat(samples)

        p_init, p_edge = self.state_prior_log_potentials(lengths, ankle.shape[1])
        frame, emission_edge = self.foot_contact_emission(ankle, toe, geometry, lengths)
        log_z = chain_log_partition(p_init, frame, p_edge + emission_edge, lengths)

        mean_contact_log_score = log_z.reshape(samples, batch_size).mean(0)
        return mean_contact_log_score - geometry_kl
