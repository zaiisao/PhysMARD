from enum import IntEnum

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from .geometry import load_support_geometry

FPS = 20.0
ENCODER_FEATURES = (
    "ankle_height", "toe_height", "foot_pitch", "ankle_speed", "toe_speed",
)
NUM_ENCODER_FEATURES = len(ENCODER_FEATURES)
GEOMETRY_DIM = 6
GEOMETRY_POSTERIOR_OUTPUT_SIZE = GEOMETRY_DIM + GEOMETRY_DIM * (GEOMETRY_DIM + 1) // 2

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
    return tuple(tuple(row) for row in codes)


def foot_frame(ankle, toe):
    """Unit `along` and `normal` axes, foot length and pitch per frame. Requires a non-vertical foot."""
    d = toe - ankle
    length = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
    along = d / length
    up = torch.zeros_like(along)
    up[..., 1] = 1.0
    normal = up - along * along[..., 1:2]
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    return along, normal, length, torch.asin(along[..., 1:2])


def support_point_positions(ankle, toe, geom):
    """World positions of heel, pad, tip [B,T,3,3]. geom [B,6] = (along, normal) per point in foot lengths."""
    along, normal, length, _ = foot_frame(ankle, toe)
    coeff = geom.view(-1, 1, 3, 2)
    offset = coeff[..., 0:1] * along[:, :, None] + coeff[..., 1:2] * normal[:, :, None]
    return ankle[:, :, None] + length[:, :, None] * offset


def flat_pitch(geom):
    """Pitch of the ankle->toe line at which heel and pad are level.

    A point (a, b) sits at height L (a sin(pitch) + b cos(pitch)) above the ankle, so heel and pad are level when
    tan(pitch) = -(b_heel - b_pad) / (a_heel - a_pad).
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


# Scores support height and pitch per frame, and support-point speed between frames.
class FootContactEmission(nn.Module):
    """Gaussian observation factors for support height, pitch, and horizontal motion."""

    def __init__(self, init_air_mean, init_pitch_mean, floor_height):
        super().__init__()

        self.air_mean = nn.Parameter(torch.tensor(init_air_mean))
        self.pitch_mean = nn.Parameter(torch.tensor(init_pitch_mean))

        self.raw_contact_sd = nn.Parameter(torch.zeros(()))
        self.raw_air_sd = nn.Parameter(torch.zeros(()))
        self.raw_flat_pitch_sd = nn.Parameter(torch.zeros(()))
        self.raw_pitch_sd = nn.Parameter(torch.zeros(()))
        self.raw_stick_speed_sd = nn.Parameter(torch.zeros(()))
        self.raw_slide_speed_sd = nn.Parameter(torch.zeros(()))
        self.raw_free_speed_sd = nn.Parameter(torch.zeros(()))

        self.floor_height = float(floor_height)

        self.support_point = tuple(SUPPORT_POINT_BY_STATE.values())
        self.is_contact = tuple(state != FootState.AIRBORNE for state in STATE_ORDER)
        self.is_flat = tuple(state in FLAT_STATES for state in STATE_ORDER)
        self.step_codes = step_model_codes()

    def forward(self, ankle, toe, geom, lengths):
        frames = ankle.shape[1]
        support_point = torch.tensor(self.support_point, device=ankle.device)
        step_codes = torch.tensor(self.step_codes, device=ankle.device)
        is_contact = torch.tensor(self.is_contact, device=ankle.device)
        is_flat = torch.tensor(self.is_flat, device=ankle.device)

        points = support_point_positions(ankle, toe, geom)
        _, _, _, pitch = foot_frame(ankle, toe)

        support_height = points[..., 1][..., support_point]
        floor = ankle.new_tensor(self.floor_height).view(1, 1, 1)
        contact_height = Normal(loc=floor, scale=F.softplus(self.raw_contact_sd))
        airborne_height = Normal(loc=floor + self.air_mean, scale=F.softplus(self.raw_air_sd))
        height_score = torch.where(
            is_contact,
            contact_height.log_prob(support_height),
            airborne_height.log_prob(support_height),
        )

        flat_pitch_dist = Normal(
            loc=flat_pitch(geom)[:, None, None],
            scale=F.softplus(self.raw_flat_pitch_sd),
        )
        other_pitch_dist = Normal(
            loc=self.pitch_mean,
            scale=F.softplus(self.raw_pitch_sd),
        )

        pitch_score = torch.where(
            is_flat,
            flat_pitch_dist.log_prob(pitch),
            other_pitch_dist.log_prob(pitch),
        )
        frame_score = height_score + pitch_score

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

        destination_points = support_point[None, :].expand(NUM_STATES, -1)
        transition_score = step_scores[:, :, destination_points, step_codes]

        valid_frames = (
            torch.arange(frames, device=ankle.device)[None] < lengths[:, None]
        )[..., None]
        valid_steps = (
            torch.arange(frames - 1, device=ankle.device)[None] < (lengths - 1)[:, None]
        )[..., None, None]

        return frame_score * valid_frames, transition_score * valid_steps

# Predicts a Gaussian over foot support geometry from ankle/toe height, pitch, and speed.
class GeometryPosterior(nn.Module):
    """Encoder and full-covariance q(g|x,side), parameterized relative to the geometry prior."""

    def __init__(self, hidden):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(NUM_ENCODER_FEATURES, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
        )
        self.cont_head = nn.Linear(hidden, GEOMETRY_POSTERIOR_OUTPUT_SIZE)

        nn.init.zeros_(self.cont_head.weight)
        nn.init.zeros_(self.cont_head.bias)

    def forward(self, inputs, lengths):
        frame_ids = torch.arange(inputs.shape[1], device=inputs.device)
        valid = frame_ids.unsqueeze(0) < lengths.unsqueeze(1)

        features = self.encoder(inputs.transpose(1, 2)).transpose(1, 2)
        pooled = (features * valid[..., None]).sum(1) / lengths[:, None]

        return self.cont_head(pooled)

    def sample_and_kl(self, raw, prior_mean, prior_covariance, sample, eps=None):
        eta_mean, tri_raw = raw[:, :GEOMETRY_DIM], raw[:, GEOMETRY_DIM:]
        tri_idx = torch.tril_indices(GEOMETRY_DIM, GEOMETRY_DIM, device=raw.device)
        diagonal = tri_idx[0] == tri_idx[1]
        tri_values = torch.where(diagonal[None], tri_raw.exp(), tri_raw)
        eta_chol = raw.new_zeros((raw.shape[0], GEOMETRY_DIM, GEOMETRY_DIM))
        eta_chol[:, tri_idx[0], tri_idx[1]] = tri_values

        prior_chol = torch.linalg.cholesky(prior_covariance)
        posterior_mean = prior_mean + torch.einsum('bij,bj->bi', prior_chol, eta_mean)
        posterior_chol = prior_chol @ eta_chol

        posterior = MultivariateNormal(posterior_mean, scale_tril=posterior_chol)
        if sample and eps is not None:
            geom = posterior_mean + torch.einsum("bij,bj->bi", posterior_chol, eps)
        else:
            geom = posterior.rsample() if sample else posterior_mean

        standard_posterior = MultivariateNormal(eta_mean, scale_tril=eta_chol)
        standard_prior = MultivariateNormal(
            torch.zeros_like(eta_mean),
            torch.eye(eta_mean.shape[-1], device=eta_mean.device, dtype=eta_mean.dtype),
        )
        kl = kl_divergence(standard_posterior, standard_prior)

        return geom, kl


class StructuredFootVAE(nn.Module):
    """HumanML3D structured foot-contact VAE with normalized geometry encoder inputs."""

    def __init__(self, input_mean, input_sd, floor_height, init_air_mean,
                 init_pitch_mean, hidden):
        super().__init__()

        if len(input_mean) != NUM_ENCODER_FEATURES or len(input_sd) != NUM_ENCODER_FEATURES:
            raise ValueError(f"Expected {NUM_ENCODER_FEATURES} encoder statistics: {', '.join(ENCODER_FEATURES)}")

        # Learned initial and transition probabilities over the seven contact states.
        self.prior_init_logits = nn.Parameter(torch.zeros(NUM_STATES))
        self.prior_transition_logits = nn.Parameter(torch.zeros(NUM_STATES, NUM_STATES))

        self.geometry_posterior = GeometryPosterior(hidden)
        self.foot_contact_emission = FootContactEmission(init_air_mean, init_pitch_mean, floor_height)

        geometry = load_support_geometry()
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
        """Five per-frame foot features: ankle/toe height, pitch, and ankle/toe speed."""
        _, _, _, pitch = foot_frame(ankle, toe)

        def speed(p):
            v = FPS * torch.linalg.vector_norm(p[:, 1:] - p[:, :-1], dim=-1, keepdim=True)
            return torch.cat([v, v[:, -1:]], 1)

        ankle_height, toe_height = ankle[..., 1:2], toe[..., 1:2]
        ankle_speed, toe_speed = speed(ankle), speed(toe)

        return ankle_height, toe_height, pitch, ankle_speed, toe_speed

    def encode(self, ankle, toe, lengths):
        """Return geometry-posterior parameters pooled over valid frames."""
        features = torch.cat(self.encoder_inputs(ankle, toe), dim=-1)
        features = (features - self.encoder_input_mean) / self.encoder_input_sd
        return self.geometry_posterior(features, lengths)

    def sample_geometry(self, ankle, toe, lengths, foot_sides, sample):
        """Draw q(g|x,side) and compute its KL to the side-specific prior."""
        raw = self.encode(ankle, toe, lengths)

        prior_mean = self.geometry_mean[foot_sides]
        prior_covariance = self.geometry_cov[foot_sides]

        return self.geometry_posterior.sample_and_kl(
            raw, prior_mean, prior_covariance, sample
        )

    def forward(self, ankle, toe, lengths, foot_sides, sample=True):
        """Return the per-sequence ELBO used for training."""
        geom, kl_g = self.sample_geometry(
            ankle, toe, lengths, foot_sides, sample
        )
        p_init, p_edge = self.state_prior_log_potentials(lengths, ankle.shape[1])

        frame, emission_edge = self.foot_contact_emission(ankle, toe, geom, lengths)
        transition_log_potentials = p_edge + emission_edge

        log_lik = chain_log_partition(
            p_init, frame, transition_log_potentials, lengths
        )
        # ELBO = expected HMM log score under q(g|x) minus KL(q(g|x) || p(g)).
        elbo = log_lik - kl_g

        return elbo
