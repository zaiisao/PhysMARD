"""Structured foot-contact model with an amortized geometry posterior.

Observation: one foot's world ankle/toe positions x_f=(ankle_f,toe_f) in R^6, in metres, Y-up, sampled at 20 Hz.
Latents: one sequence-shared six-coordinate foot geometry g with a side-specific Gaussian prior; and a seven-state
Markov chain s_1:T. The floor height is a fixed buffer, not a latent variable.

The code constructs normalized Gaussian log factors for selected transformed features: support-point height, pitch,
and horizontal support-point displacements. It omits yaw, toe-ankle length, and the first support point's horizontal
anchor. Therefore log_likelihood is the exact state-marginalized score for this reduced feature model, not a complete
normalized density over the raw ankle/toe trajectory. The selected factors also assume conditional independence,
including independent horizontal-step errors over time.

Given geometry, the state-chain partition function and marginals are computed exactly by log-semiring forward-backward.
Only geometry uses an amortized full-covariance Gaussian q_phi(g|x,u,side); there is no learned discrete-state encoder.
Optional geometry marginalization averages state marginals over geometry draws for reporting. Optional geometry_scale
is post-hoc uncertainty scaling and does not alter the ELBO or trained posterior parameters.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from geometry import load_support_geometry

FPS = 20.0
NUM_STATES = 7
STATE_REGION = (-1, 0, 0, 1, 1, 2, 2)   # -1 airborne, 0 flat, 1 forefoot, 2 tip
STATE_STICK = (-1, 1, 0, 1, 0, 1, 0)    # 1 stick, 0 slide
HEEL, PAD, TIP = 0, 1, 2
SUPPORT_POINT = (PAD, PAD, PAD, PAD, PAD, TIP, TIP)
FLAT_STATES = (1, 2)
FREE, STICK, SLIDE = 0, 1, 2


def gaussian_logpdf(x, mean, sd):
    return -0.5 * ((x - mean) / sd).square() - torch.log(sd) - 0.5 * math.log(2 * math.pi)


def step_model_codes():
    """[7, 7]: how the support point's horizontal step is scored for a transition state i -> state j."""
    codes = torch.full((NUM_STATES, NUM_STATES), FREE, dtype=torch.long)
    for i in range(NUM_STATES):
        for j in range(NUM_STATES):
            if STATE_REGION[i] >= 0 and STATE_REGION[i] == STATE_REGION[j]:
                codes[i, j] = STICK if STATE_STICK[i] == 1 and STATE_STICK[j] == 1 else SLIDE
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
    return torch.atan(-(g[:, HEEL, 1] - g[:, PAD, 1]) / (g[:, HEEL, 0] - g[:, PAD, 0]))


def chain_marginals_sequential(init, frame, edge, lengths):
    """Reference implementation: exact forward-backward with a Python loop over frames (slow on GPU). init [B,K], frame [B,T,K], edge [B,T-1,K,K] log-potentials.

    Returns log partition [B], state marginals [B,T,K], pair marginals [B,T-1,K,K].
    """
    b, t, k = frame.shape
    alpha = [init + frame[:, 0]]
    log_z = torch.logsumexp(alpha[0], -1)
    alpha[0] = alpha[0] - log_z[:, None]
    for f in range(1, t):
        raw = torch.logsumexp(alpha[-1][:, :, None] + edge[:, f - 1], 1) + frame[:, f]
        c = torch.logsumexp(raw, -1)
        valid = lengths > f
        alpha.append(torch.where(valid[:, None], raw - c[:, None], alpha[-1]))
        log_z = log_z + torch.where(valid, c, torch.zeros_like(c))
    alpha = torch.stack(alpha, 1)

    beta = [frame.new_zeros(b, k)]
    for f in range(t - 2, -1, -1):
        val = torch.logsumexp(edge[:, f] + frame[:, f + 1, None, :] + beta[0][:, None, :], -1)
        val = val - torch.logsumexp(val, -1, keepdim=True)
        beta.insert(0, torch.where((lengths > f + 1)[:, None], val, torch.zeros_like(val)))
    beta = torch.stack(beta, 1)

    frame_valid = torch.arange(t, device=frame.device)[None] < lengths[:, None]
    gamma = torch.softmax(alpha + beta, -1) * frame_valid[..., None]
    pair = alpha[:, :-1, :, None] + edge + frame[:, 1:, None, :] + beta[:, 1:, None, :]
    xi = torch.softmax(pair.flatten(-2), -1).reshape(b, t - 1, k, k)
    edge_valid = torch.arange(t - 1, device=frame.device)[None] < (lengths - 1)[:, None]
    return log_z, gamma, xi * edge_valid[..., None, None]


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


def chain_marginals(init, frame, edge, lengths):
    """Exact forward-backward by parallel prefix scans: O(log T) sequential steps instead of T.

    init [B,K], frame [B,T,K], edge [B,T-1,K,K] log-potentials (T >= 2).
    Returns log partition [B], state marginals [B,T,K], pair marginals [B,T-1,K,K]; same values as
    chain_marginals_sequential. Transition t-1 -> t has matrix M_t[i, j] = edge[t-1][i, j] + frame[t][j]; steps past a
    sequence's length use the identity matrix (0 on the diagonal, -inf off it), which leaves every quantity unchanged.
    """
    b, t, k = frame.shape
    eye = torch.full((k, k), -1e9, dtype=frame.dtype, device=frame.device)
    eye.fill_diagonal_(0.0)
    step_valid = (torch.arange(1, t, device=frame.device)[None] < lengths[:, None])[..., None, None]
    m = torch.where(step_valid, edge + frame[:, 1:, None, :], eye)                      # [B, T-1, K, K]

    alpha0 = init + frame[:, 0]                                                          # [B, K]
    prefix = log_prefix_scan(m)                                                          # prefix[:, s] = M_1 ... M_{s+1}
    alpha = torch.cat([alpha0[:, None], torch.logsumexp(alpha0[:, None, :, None] + prefix, dim=-2)], dim=1)   # [B, T, K]
    log_z = torch.logsumexp(alpha[:, -1], dim=-1)

    # Suffix products S[:, s] = M_{s+1} ... M_{T-1}, from a prefix scan of the reversed, transposed sequence.
    reversed_scan = log_prefix_scan(m.flip(1).transpose(-1, -2))
    suffix = reversed_scan.flip(1).transpose(-1, -2)
    beta = torch.cat([torch.logsumexp(suffix, dim=-1), frame.new_zeros(b, 1, k)], dim=1)                      # [B, T, K]

    frame_valid = torch.arange(t, device=frame.device)[None] < lengths[:, None]
    gamma = torch.softmax(alpha + beta, -1) * frame_valid[..., None]
    pair = alpha[:, :-1, :, None] + edge + frame[:, 1:, None, :] + beta[:, 1:, None, :]
    xi = torch.softmax(pair.flatten(-2), -1).reshape(b, t - 1, k, k)
    edge_valid = torch.arange(t - 1, device=frame.device)[None] < (lengths - 1)[:, None]
    return log_z, gamma, xi * edge_valid[..., None, None]


def kl_diag_to_full(q_mean, q_var, p_mean, p_cov):
    """KL( N(q_mean, diag q_var) || N(p_mean, p_cov) ) per row."""
    p_inv = torch.linalg.inv(p_cov)
    diff = (p_mean - q_mean)[..., None]
    trace = (torch.diagonal(p_inv, dim1=-2, dim2=-1) * q_var).sum(-1)
    mahalanobis = (diff.transpose(-1, -2) @ p_inv @ diff)[..., 0, 0]
    return 0.5 * (trace + mahalanobis - q_mean.shape[-1] + torch.logdet(p_cov) - torch.log(q_var).sum(-1))


class StructuredFootVAE(nn.Module):
    """Full-covariance fixed-floor candidate. Learned: observation scales, state prior and q(g|x,u)."""

    def __init__(self, context_dim, geometry=None, hidden=64,
                 init_contact_sd=0.01, init_air_mean=0.10, init_air_sd=0.08,
                 init_flat_sd=0.05, init_pitch_mean=-0.3, init_pitch_sd=0.5,
                 init_stick_sd=0.03, init_slide_sd=0.5, init_free_sd=1.0, fixed_floor_height=0.0):
        super().__init__()

        # --- Fixed structure.
        self.register_buffer("support_point", torch.tensor(SUPPORT_POINT))
        self.register_buffer("step_codes", step_model_codes())
        self.register_buffer("is_contact", torch.tensor([r >= 0 for r in STATE_REGION]))
        self.register_buffer("is_flat", torch.tensor([s in FLAT_STATES for s in range(NUM_STATES)]))

        # --- Foot geometry prior from SMPL body shapes (per foot side).
        geometry = geometry or load_support_geometry()
        sides = ('left', 'right')
        self.register_buffer("geometry_mean", torch.tensor([geometry[s]['mean'] for s in sides]))       # [2, 6]
        self.register_buffer("geometry_cov", torch.tensor([geometry[s]['covariance'] for s in sides]))  # [2, 6, 6]

        # --- Fixed support plane in the declared world coordinate frame.
        self.register_buffer("fixed_floor_height", torch.tensor(float(fixed_floor_height)))

        # --- Observation model scales (learned).
        def log_param(value):
            return nn.Parameter(torch.tensor(math.log(value)))
        self.log_contact_sd = log_param(init_contact_sd)      # support point on the floor (m)
        self.air_mean = nn.Parameter(torch.tensor(init_air_mean))
        self.log_air_sd = log_param(init_air_sd)              # support point height when airborne (m)
        self.log_flat_sd = log_param(init_flat_sd)            # pitch spread around the flat pitch (rad)
        self.pitch_mean = nn.Parameter(torch.tensor(init_pitch_mean))
        self.log_pitch_sd = log_param(init_pitch_sd)          # pitch when not flat (rad)
        self.log_stick_sd = log_param(init_stick_sd)          # support point speed when sticking (m/s)
        self.log_slide_sd = log_param(init_slide_sd)          # ... when sliding
        self.log_free_sd = log_param(init_free_sd)            # ... when unsupported or changing support

        # --- State prior, conditioned on context u; starts uniform.
        self.prior_init = nn.Linear(context_dim, NUM_STATES)
        self.prior_edge = nn.Sequential(nn.Linear(2 * context_dim, hidden), nn.Tanh(), nn.Linear(hidden, NUM_STATES ** 2))
        for layer in (self.prior_init, self.prior_edge[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # --- Encoder for q(g | x, u), full covariance in prior-whitened coordinates (q starts at prior).
        # Per-frame inputs: ankle height, toe height, pitch, ankle speed, toe speed, context u.
        self.encoder = nn.Sequential(nn.Conv1d(5 + context_dim, hidden, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU())
        self.cont_head = nn.Linear(hidden, 27)
        nn.init.zeros_(self.cont_head.weight)
        nn.init.zeros_(self.cont_head.bias)

    # ---- observation model ----------------------------------------------------------------
    def observation_log_potentials(self, ankle, toe, geom, lengths):
        """Per-frame log-density [B,T,7] (support height + pitch) and per-step log-density [B,T-1,7,7]."""
        t = ankle.shape[1]
        points = support_point_positions(ankle, toe, geom)                    # [B,T,3 points,3]
        _, _, _, pitch = foot_frame(ankle, toe)                               # [B,T,1]

        support_y = points[..., 1][..., self.support_point]                   # [B,T,7]
        h = self.fixed_floor_height.to(device=ankle.device, dtype=ankle.dtype).view(1, 1, 1)
        height_lp = torch.where(self.is_contact,
                                gaussian_logpdf(support_y, h, self.log_contact_sd.exp()),
                                gaussian_logpdf(support_y, h + self.air_mean, self.log_air_sd.exp()))
        pitch_lp = torch.where(self.is_flat,
                               gaussian_logpdf(pitch, flat_pitch(geom)[:, None, None], self.log_flat_sd.exp()),
                               gaussian_logpdf(pitch, self.pitch_mean, self.log_pitch_sd.exp()))
        frame = height_lp + pitch_lp

        # Horizontal step of the destination state's support point (same material point on both frames).
        step = FPS * (points[:, 1:] - points[:, :-1])[..., (0, 2)]           # [B,T-1,3 points,2]
        by_model = torch.stack([gaussian_logpdf(step, 0.0, sd).sum(-1) for sd in
                                (self.log_free_sd.exp(), self.log_stick_sd.exp(), self.log_slide_sd.exp())], -1)
        destination_point = self.support_point[None, :].expand(NUM_STATES, -1)  # [7 from, 7 to]
        edge = by_model[:, :, destination_point, self.step_codes]              # [B,T-1,7,7]

        valid = (torch.arange(t, device=ankle.device)[None] < lengths[:, None])[..., None]
        edge_valid = (torch.arange(t - 1, device=ankle.device)[None] < (lengths - 1)[:, None])[..., None, None]
        return frame * valid, edge * edge_valid

    def log_likelihood(self, u, ankle, toe, geom, lengths):
        """Exact state-marginalized score at fixed floor height and exact state posterior given g."""
        b, t, _ = u.shape
        valid = torch.arange(t, device=u.device)[None] < lengths[:, None]
        init_context = (u * valid[..., None]).sum(1) / lengths[:, None]
        p_init = F.log_softmax(self.prior_init(init_context), -1)
        pairs = torch.cat([u[:, :-1], u[:, 1:]], -1)
        p_edge = F.log_softmax(self.prior_edge(pairs).reshape(b, t - 1, NUM_STATES, NUM_STATES), -1)
        p_edge = p_edge * (torch.arange(t - 1, device=u.device)[None] < (lengths - 1)[:, None])[..., None, None]
        frame, edge = self.observation_log_potentials(ankle, toe, geom, lengths)
        return chain_marginals(p_init, frame, p_edge + edge, lengths)

    # ---- ELBO ------------------------------------------------------------------------------
    def encoder_inputs(self, u, ankle, toe):
        """Per-frame encoder inputs [B,T,5+context]: ankle height, toe height, pitch, ankle speed, toe speed, u."""
        _, _, _, pitch = foot_frame(ankle, toe)

        def speed(p):
            v = FPS * torch.linalg.vector_norm(p[:, 1:] - p[:, :-1], dim=-1, keepdim=True)
            return torch.cat([v, v[:, -1:]], 1)

        return torch.cat([ankle[..., 1:2], toe[..., 1:2], pitch, speed(ankle), speed(toe), u], -1)

    def encode(self, u, ankle, toe, lengths):
        """Raw q(g | x) parameters [B,27]: per-frame conv features averaged over valid frames."""
        valid = torch.arange(u.shape[1], device=u.device)[None] < lengths[:, None]
        feats = self.encoder(self.encoder_inputs(u, ankle, toe).transpose(1, 2)).transpose(1, 2)
        pooled = (feats * valid[..., None]).sum(1) / lengths[:, None]
        return self.cont_head(pooled)

    def forward(self, u, ankle, toe, lengths, foot_sides, sample=True,
                marginalize_geometry=False, geometry_samples=32, geometry_scale=1.0):
        """Every sequence has at least 2 frames; feet are never vertical.

        By default, `gamma` and `xi` are conditional on the sampled geometry
        used by the ELBO (or on the posterior mean when sample=False). Set
        `marginalize_geometry=True` to estimate state marginals under q(g|x,side)
        by averaging conditional chain marginals over shared sequence-level
        geometry draws. The ELBO remains a single-sample estimate.
        """
        b, t = u.shape[:2]
        if geometry_scale <= 0:
            raise ValueError("geometry_scale must be positive")
        valid = torch.arange(t, device=u.device)[None] < lengths[:, None]
        g_mean, g_cov = self.geometry_mean[foot_sides], self.geometry_cov[foot_sides]

        # q(g | x, u), relative to the geometry prior.
        raw = self.encode(u, ankle, toe, lengths)
        q_eta_mean, tri_raw = raw[:, :6], raw[:, 6:]
        tri_idx = torch.tril_indices(6, 6, device=raw.device)
        diag_mask = tri_idx[0] == tri_idx[1]
        tri_values = torch.where(diag_mask[None], tri_raw.exp(), tri_raw)
        q_eta_chol = raw.new_zeros((b, 6, 6))
        q_eta_chol[:, tri_idx[0], tri_idx[1]] = tri_values

        # Define q in prior-whitened coordinates: eta ~ N(q_eta_mean, L L^T),
        # g = prior_mean + prior_cholesky @ eta. This is a full covariance q(g|x).
        prior_chol = torch.linalg.cholesky(g_cov)
        q_g_mean = g_mean + torch.einsum('bij,bj->bi', prior_chol, q_eta_mean)
        q_g_chol = prior_chol @ q_eta_chol
        eps = torch.randn_like(q_eta_mean) if sample else torch.zeros_like(q_eta_mean)
        geom = q_g_mean + torch.einsum('bij,bj->bi', q_g_chol, eps)

        # KL[N(m, L L^T) || N(0,I)] in whitened coordinates.
        q_eta_cov = q_eta_chol @ q_eta_chol.transpose(-1, -2)
        kl_g = 0.5 * (q_eta_mean.square().sum(-1)
                      + torch.diagonal(q_eta_cov, dim1=-2, dim2=-1).sum(-1)
                      - 6 - 2 * torch.log(torch.diagonal(q_eta_chol, dim1=-2, dim2=-1)).sum(-1))
        log_lik, gamma, xi = self.log_likelihood(u, ankle, toe, geom, lengths)
        if marginalize_geometry:
            if geometry_samples < 1:
                raise ValueError("geometry_samples must be at least 1")
            # One joint geometry draw per sequence, shared across every frame.
            # E_q[ p(s_f | x,g) ] gives the geometry-marginalized state marginal.
            eps_mc = torch.randn(geometry_samples, b, 6, device=u.device, dtype=u.dtype)
            geom_mc = q_g_mean[None] + geometry_scale * torch.einsum('bij,nbj->nbi', q_g_chol, eps_mc)
            flat_geom = geom_mc.reshape(geometry_samples * b, 6)
            mc_u = u.unsqueeze(0).expand(geometry_samples, *u.shape).reshape(geometry_samples * b, *u.shape[1:])
            mc_ankle = ankle.unsqueeze(0).expand(geometry_samples, *ankle.shape).reshape(geometry_samples * b, *ankle.shape[1:])
            mc_toe = toe.unsqueeze(0).expand(geometry_samples, *toe.shape).reshape(geometry_samples * b, *toe.shape[1:])
            mc_lengths = lengths.unsqueeze(0).expand(geometry_samples, b).reshape(-1)
            mc_log_lik, mc_gamma, mc_xi = self.log_likelihood(
                mc_u, mc_ankle, mc_toe, flat_geom, mc_lengths)
            mc_gamma = mc_gamma.reshape(geometry_samples, b, t, NUM_STATES)
            mc_xi = mc_xi.reshape(geometry_samples, b, max(t - 1, 0), NUM_STATES, NUM_STATES)
            # Keep log_likelihood and ELBO semantics tied to the original ELBO draw.
            gamma = mc_gamma.mean(0)
            xi = mc_xi.mean(0)
        q_g_cov = q_g_chol @ q_g_chol.transpose(-1, -2)
        calibrated_chol = geometry_scale * q_g_chol
        calibrated_cov = geometry_scale ** 2 * q_g_cov
        return dict(elbo=log_lik - kl_g, log_likelihood=log_lik, gamma=gamma, xi=xi,
                    geometry=geom, geometry_mean=q_g_mean, geometry_cholesky=q_g_chol,
                    geometry_covariance=q_g_cov, geometry_calibrated_cholesky=calibrated_chol,
                    geometry_calibrated_covariance=calibrated_cov,
                    geometry_scale=torch.full((b,),float(geometry_scale),device=u.device,dtype=u.dtype),
                    floor_height=self.fixed_floor_height.expand(b))
