from enum import IntEnum

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from geometry import load_support_geometry

FPS = 20.0

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
STICKING_SLIDING_GROUPS = (
    (FootState.FLAT_STICKING, FootState.FLAT_SLIDING),
    (FootState.FOREFOOT_STICKING, FootState.FOREFOOT_SLIDING),
    (FootState.TOE_TIP_STICKING, FootState.TOE_TIP_SLIDING),
)
STATE_ORDER = tuple(FootState)

def step_model_codes():
    """[7, 7]: how the support point's horizontal step is scored for a transition state i -> state j."""
    codes = torch.full(
        (NUM_STATES, NUM_STATES), int(StepModel.FREE), dtype=torch.long
    )
    for sticking, sliding in STICKING_SLIDING_GROUPS:
        codes[sticking, sticking] = StepModel.STICK
        codes[sticking, sliding] = StepModel.SLIDE
        codes[sliding, sticking] = StepModel.SLIDE
        codes[sliding, sliding] = StepModel.SLIDE
    return codes


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

    heel_normal = g[:, SupportPoint.HEEL, 1]
    pad_normal = g[:, SupportPoint.PAD, 1]
    heel_along = g[:, SupportPoint.HEEL, 0]
    pad_along = g[:, SupportPoint.PAD, 0]

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

        state_support_points = torch.tensor([
            int(SUPPORT_POINT_BY_STATE[state]) for state in STATE_ORDER
        ])
        state_step_codes = step_model_codes()
        contact_state_mask = torch.tensor([
            state != FootState.AIRBORNE for state in STATE_ORDER
        ])
        flat_state_mask = torch.tensor([
            state in (FootState.FLAT_STICKING, FootState.FLAT_SLIDING)
            for state in STATE_ORDER
        ])
        fixed_floor_height = torch.tensor(float(floor_height))

        self.register_buffer("support_point", state_support_points)
        self.register_buffer("step_codes", state_step_codes)
        self.register_buffer("is_contact", contact_state_mask)
        self.register_buffer("is_flat", flat_state_mask)
        self.register_buffer("floor_height", fixed_floor_height)

    def forward(self, ankle, toe, geom, lengths):
        frames = ankle.shape[1]
        points = support_point_positions(ankle, toe, geom)
        _, _, _, pitch = foot_frame(ankle, toe)

        support_height = points[..., 1][..., self.support_point]
        floor = self.floor_height.to(device=ankle.device, dtype=ankle.dtype).view(1, 1, 1)
        contact_height = Normal(loc=floor, scale=F.softplus(self.raw_contact_sd))
        airborne_height = Normal(loc=floor + self.air_mean, scale=F.softplus(self.raw_air_sd))
        height_score = torch.where(
            self.is_contact,
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
            self.is_flat,
            flat_pitch_dist.log_prob(pitch),
            other_pitch_dist.log_prob(pitch),
        )
        frame_score = height_score + pitch_score

        horizontal_step = FPS * (points[:, 1:] - points[:, :-1])[..., (0, 2)]
        speed_sds = (
            F.softplus(self.raw_free_speed_sd),
            F.softplus(self.raw_stick_speed_sd),
            F.softplus(self.raw_slide_speed_sd),
        )
        step_scores = torch.stack([
            Normal(loc=0.0, scale=sd).log_prob(horizontal_step).sum(-1)
            for sd in speed_sds
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
    """Encoder and full-covariance q(g|x,u,side), parameterized relative to the geometry prior."""

    def __init__(self, context_dim, hidden):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(5 + context_dim, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
        )
        self.cont_head = nn.Linear(hidden, 27)
        nn.init.zeros_(self.cont_head.weight)
        nn.init.zeros_(self.cont_head.bias)

    def forward(self, inputs, lengths):
        valid = torch.arange(inputs.shape[1], device=inputs.device)[None] < lengths[:, None]
        features = self.encoder(inputs.transpose(1, 2)).transpose(1, 2)
        pooled = (features * valid[..., None]).sum(1) / lengths[:, None]
        return self.cont_head(pooled)

    def sample_and_kl(self, raw, prior_mean, prior_covariance, sample):
        eta_mean, tri_raw = raw[:, :6], raw[:, 6:]
        tri_idx = torch.tril_indices(6, 6, device=raw.device)
        diagonal = tri_idx[0] == tri_idx[1]
        tri_values = torch.where(diagonal[None], tri_raw.exp(), tri_raw)
        eta_chol = raw.new_zeros((raw.shape[0], 6, 6))
        eta_chol[:, tri_idx[0], tri_idx[1]] = tri_values

        prior_chol = torch.linalg.cholesky(prior_covariance)
        posterior_mean = prior_mean + torch.einsum('bij,bj->bi', prior_chol, eta_mean)
        posterior_chol = prior_chol @ eta_chol

        posterior = MultivariateNormal(posterior_mean, scale_tril=posterior_chol)
        geom = posterior.rsample() if sample else posterior_mean
        standard_posterior = MultivariateNormal(eta_mean, scale_tril=eta_chol)
        standard_prior = MultivariateNormal(
            torch.zeros_like(eta_mean),
            torch.eye(eta_mean.shape[-1], device=eta_mean.device, dtype=eta_mean.dtype),
        )
        kl = kl_divergence(standard_posterior, standard_prior)
        return geom, kl


class StructuredFootVAE(nn.Module):
    """Full-covariance fixed-floor candidate. Learned: observation scales, state prior and q(g|x,u)."""

    def __init__(self, context_dim, geometry=None, hidden=64,
                 init_air_mean=0.10, init_pitch_mean=-0.3, fixed_floor_height=0.0):
        super().__init__()

        self.prior_init = nn.Linear(context_dim, NUM_STATES)
        self.prior_edge = nn.Sequential(nn.Linear(2 * context_dim, hidden), nn.Tanh(), nn.Linear(hidden, NUM_STATES ** 2))
        for layer in (self.prior_init, self.prior_edge[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.geometry_posterior_network = GeometryPosterior(context_dim, hidden)
        self.emission = FootContactEmission(
            init_air_mean, init_pitch_mean, fixed_floor_height
        )

        geometry = geometry or load_support_geometry()
        sides = ('left', 'right')
        geometry_mean = torch.tensor([geometry[side]['mean'] for side in sides])
        geometry_covariance = torch.tensor([
            geometry[side]['covariance'] for side in sides
        ])
        self.register_buffer("geometry_mean", geometry_mean)
        self.register_buffer("geometry_cov", geometry_covariance)

    def state_prior_log_potentials(self, u, lengths):
        """Context-conditioned initial and transition log-probabilities [B,K], [B,T-1,K,K]."""
        b, t, _ = u.shape
        valid = torch.arange(t, device=u.device)[None] < lengths[:, None]
        init_context = (u * valid[..., None]).sum(1) / lengths[:, None]
        p_init = F.log_softmax(self.prior_init(init_context), -1)
        pairs = torch.cat([u[:, :-1], u[:, 1:]], -1)
        p_edge = F.log_softmax(self.prior_edge(pairs).reshape(b, t - 1, NUM_STATES, NUM_STATES), -1)
        p_edge = p_edge * (torch.arange(t - 1, device=u.device)[None] < (lengths - 1)[:, None])[..., None, None]
        return p_init, p_edge

    def encoder_inputs(self, u, ankle, toe):
        """Per-frame encoder inputs [B,T,5+context]: ankle height, toe height, pitch, ankle speed, toe speed, u."""
        _, _, _, pitch = foot_frame(ankle, toe)

        def speed(p):
            v = FPS * torch.linalg.vector_norm(p[:, 1:] - p[:, :-1], dim=-1, keepdim=True)
            return torch.cat([v, v[:, -1:]], 1)

        return torch.cat([ankle[..., 1:2], toe[..., 1:2], pitch, speed(ankle), speed(toe), u], -1)

    def encode(self, u, ankle, toe, lengths):
        """Raw q(g | x) parameters [B,27]: per-frame conv features averaged over valid frames."""
        return self.geometry_posterior_network(
            self.encoder_inputs(u, ankle, toe), lengths
        )

    def sample_geometry(self, u, ankle, toe, lengths, foot_sides, sample):
        """Draw q(g|x,u,side) and compute its KL to the side-specific prior."""
        raw = self.encode(u, ankle, toe, lengths)
        prior_mean = self.geometry_mean[foot_sides]
        prior_covariance = self.geometry_cov[foot_sides]
        return self.geometry_posterior_network.sample_and_kl(
            raw, prior_mean, prior_covariance, sample
        )

    def forward(self, u, ankle, toe, lengths, foot_sides, sample=True):
        """Return the per-sequence ELBO used for training."""
        geom, kl_g = self.sample_geometry(
            u, ankle, toe, lengths, foot_sides, sample
        )
        p_init, p_edge = self.state_prior_log_potentials(u, lengths)

        frame, emission_edge = self.emission(ankle, toe, geom, lengths)
        transition_log_potentials = p_edge + emission_edge

        log_lik = chain_log_partition(
            p_init, frame, transition_log_potentials, lengths
        )
        elbo = log_lik - kl_g

        return elbo
