"""Experimental text-conditioned Markov contact prior and score helpers."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

NUM_FEET, NUM_STATES = 2, 7


class TextContactPrior(nn.Module):
    """Independent per-foot, text-conditioned seven-state first-order Markov prior."""
    def __init__(self, text_dim=512, hidden_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.GELU())
        self.initial = nn.Linear(hidden_dim, NUM_FEET * NUM_STATES)
        self.transition = nn.Linear(hidden_dim, NUM_FEET * NUM_STATES * NUM_STATES)

    def log_probs(self, text_embedding):
        h = self.trunk(text_embedding)
        b = h.shape[0]
        init = self.initial(h).view(b, NUM_FEET, NUM_STATES).log_softmax(-1)
        edge = self.transition(h).view(b, NUM_FEET, NUM_STATES, NUM_STATES).log_softmax(-1)
        return init, edge

    def sample(self, text_embedding, lengths, generator=None):
        init, edge = self.log_probs(text_embedding)
        b, feet, states = init.shape
        tmax = int(lengths.max().item())
        paths = torch.zeros((b, feet, tmax), dtype=torch.long, device=text_embedding.device)
        first = torch.multinomial(init.exp().reshape(-1, states), 1, generator=generator).reshape(b, feet)
        paths[:, :, 0] = first
        for t in range(1, tmax):
            prev = paths[:, :, t - 1]
            probs = edge.exp().gather(-2, prev[..., None, None].expand(-1, -1, 1, states)).squeeze(-2)
            draw = torch.multinomial(probs.reshape(-1, states), 1, generator=generator).reshape(b, feet)
            paths[:, :, t] = torch.where((lengths > t)[:, None], draw, prev)
        return paths

    def log_prob(self, states, text_embedding, lengths=None):
        init, edge = self.log_probs(text_embedding)
        b, feet, frames = states.shape
        if lengths is None:
            lengths = torch.full((b,), frames, device=states.device, dtype=torch.long)
        first = init.gather(-1, states[..., :1]).squeeze(-1).sum(-1)
        total = first
        for t in range(1, frames):
            value = edge.gather(-2, states[..., t - 1, None, None].expand(-1, -1, 1, NUM_STATES)).squeeze(-2)
            value = value.gather(-1, states[..., t, None]).squeeze(-1).sum(-1)
            total = total + value * (lengths > t)
        return total


def hmm_marginals_and_xi(init, frame, edge, lengths):
    """Exact node and edge marginals for variable-length batched chains."""
    batch, frames, states = frame.shape
    all_gamma, all_xi = [], []
    for b in range(batch):
        n = int(lengths[b].item())
        alpha = [init[b] + frame[b, 0]]
        for t in range(1, n):
            alpha.append(torch.logsumexp(alpha[-1][:, None] + edge[b, t - 1], dim=0) + frame[b, t])
        log_z = torch.logsumexp(alpha[-1], dim=0)
        beta = [None] * n
        beta[-1] = torch.zeros_like(alpha[-1])
        for t in range(n - 2, -1, -1):
            beta[t] = torch.logsumexp(edge[b, t] + frame[b, t + 1, None, :] + beta[t + 1][None, :], dim=-1)
        gamma = torch.stack([(alpha[t] + beta[t] - log_z).exp() for t in range(n)])
        xi = torch.stack([
            (alpha[t][:, None] + edge[b, t] + frame[b, t + 1, None, :] + beta[t + 1][None, :] - log_z).exp()
            for t in range(n - 1)
        ]) if n > 1 else frame.new_zeros((0, states, states))
        if n < frames:
            gamma = torch.cat((gamma, gamma[-1:].expand(frames - n, -1)))
            xi = torch.cat((xi, xi.new_zeros((frames - n, states, states))))
        all_gamma.append(gamma); all_xi.append(xi)
    return torch.stack(all_gamma), torch.stack(all_xi)


def hmm_kl_to_text_prior(q_init, q_frame, q_edge, p_init, p_edge, lengths):
    gamma, xi = hmm_marginals_and_xi(q_init, q_frame, q_edge, lengths)
    tiny = torch.finfo(gamma.dtype).tiny
    entropy = -(gamma[:, 0] * gamma[:, 0].clamp_min(tiny).log()).sum(-1)
    cross = -(gamma[:, 0] * p_init).sum(-1)
    for t in range(q_edge.shape[1]):
        valid = lengths > t + 1
        conditional = xi[:, t] / gamma[:, t, :, None].clamp_min(tiny)
        entropy_t = -(xi[:, t] * conditional.clamp_min(tiny).log()).sum((-1, -2))
        cross_t = -(xi[:, t] * p_edge).sum((-1, -2))
        entropy = entropy + torch.where(valid, entropy_t, torch.zeros_like(entropy_t))
        cross = cross + torch.where(valid, cross_t, torch.zeros_like(cross_t))
    return cross - entropy


@torch.no_grad()
def sample_hmm_paths(init, frame, edge, lengths, generator=None):
    batch, frames, states = frame.shape
    paths = torch.zeros((batch, frames), device=frame.device, dtype=torch.long)
    for b in range(batch):
        n = int(lengths[b].item())
        alpha = [init[b] + frame[b, 0]]
        for t in range(1, n):
            alpha.append(torch.logsumexp(alpha[-1][:, None] + edge[b, t - 1], dim=0) + frame[b, t])
        paths[b, n - 1] = torch.multinomial(alpha[-1].softmax(-1), 1, generator=generator).squeeze(0)
        for t in range(n - 2, -1, -1):
            next_state = paths[b, t + 1]
            conditional = alpha[t] + edge[b, t, :, next_state]
            paths[b, t] = torch.multinomial(conditional.softmax(-1), 1, generator=generator).squeeze(0)
        if n < frames:
            paths[b, n:] = paths[b, n - 1]
    return paths


def hmm_path_log_prob(init, frame, edge, lengths, paths):
    scores = []
    for b in range(frame.shape[0]):
        n = int(lengths[b].item())
        alpha = init[b] + frame[b, 0]
        score = alpha[paths[b, 0]]
        for t in range(1, n):
            score = score + edge[b, t - 1, paths[b, t - 1], paths[b, t]] + frame[b, t, paths[b, t]]
            alpha = torch.logsumexp(alpha[:, None] + edge[b, t - 1], dim=0) + frame[b, t]
        scores.append(score - torch.logsumexp(alpha, dim=0))
    return torch.stack(scores)


def sample_geometry_prior(mean, covariance, generator=None):
    noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    return mean + torch.matmul(torch.linalg.cholesky(covariance), noise.unsqueeze(-1)).squeeze(-1)


def score_function_loss(reconstruction_log_prob, sampled_path_log_prob, baseline):
    advantage = (reconstruction_log_prob - baseline).detach()
    return -(advantage * sampled_path_log_prob).mean()
