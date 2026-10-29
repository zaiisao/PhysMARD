"""Numerical checks for the contact-CVAE HMM and geometry helpers."""
import itertools
import unittest

import torch

from models.contact_cvae import (
    TextContactPrior,
    hmm_kl_to_text_prior,
    hmm_marginals_and_xi,
    hmm_path_log_prob,
    sample_geometry_prior,
    sample_hmm_paths,
    score_function_loss,
)


def binary_chain(dtype=torch.double):
    init = torch.tensor([[0.3, -0.2]], dtype=dtype, requires_grad=True)
    edge = torch.tensor(
        [[[[0.1, -0.4], [0.2, 0.3]], [[-0.3, 0.5], [0.1, -0.1]],
          [[0.6, -0.2], [-0.1, 0.4]]]], dtype=dtype, requires_grad=True
    )
    frame = torch.tensor([[[0.2, -0.1], [-0.2, 0.6], [0.3, -0.4], [-0.1, 0.2]]],
                         dtype=dtype, requires_grad=True)
    p_init_logits = torch.tensor([[-0.2, 0.4]], dtype=dtype, requires_grad=True)
    p_edge_logits = torch.tensor([[[0.4, -0.3], [-0.2, 0.6]]], dtype=dtype, requires_grad=True)
    paths = torch.tensor(list(itertools.product(range(2), repeat=4)), dtype=torch.long)
    scores = init[0, paths[:, 0]] + frame[0, 0, paths[:, 0]]
    for t in range(1, 4):
        scores = scores + edge[0, t-1, paths[:, t-1], paths[:, t]] + frame[0, t, paths[:, t]]
    log_q = scores - torch.logsumexp(scores, dim=0)
    log_p_init = p_init_logits.log_softmax(-1)
    log_p_edge = p_edge_logits.log_softmax(-1)
    log_p = log_p_init[0, paths[:, 0]]
    for t in range(1, 4):
        log_p = log_p + log_p_edge[0, paths[:, t-1], paths[:, t]]
    return init, edge, frame, p_init_logits, p_edge_logits, paths, log_q, log_p


class ContactCVAETests(unittest.TestCase):
    def test_exact_hmm_kl_marginals_and_gradients(self):
        init, edge, frame, p_init_logits, p_edge_logits, paths, log_q, log_p = binary_chain()
        q = log_q.exp()
        exact_kl = (q * (log_q - log_p)).sum()
        got = hmm_kl_to_text_prior(
            init, frame, edge, p_init_logits.log_softmax(-1),
            p_edge_logits.log_softmax(-1), torch.tensor([4])
        )[0]
        torch.testing.assert_close(got, exact_kl, rtol=1e-13, atol=1e-13)

        gamma, xi = hmm_marginals_and_xi(init, frame, edge, torch.tensor([4]))
        exact_gamma = torch.stack([
            torch.stack([q[paths[:, t] == s].sum() for s in range(2)])
            for t in range(4)
        ])
        exact_xi = torch.stack([
            torch.stack([torch.stack([q[(paths[:, t] == i) & (paths[:, t+1] == j)].sum()
                                      for j in range(2)]) for i in range(2)])
            for t in range(3)
        ])
        torch.testing.assert_close(gamma[0], exact_gamma, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(xi[0, :3], exact_xi, rtol=1e-13, atol=1e-13)

        variables = (init, edge, frame, p_init_logits, p_edge_logits)
        exact_grad = torch.autograd.grad(exact_kl, variables, retain_graph=True)
        actual_grad = torch.autograd.grad(got, variables)
        for actual, expected in zip(actual_grad, exact_grad):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_forward_backward_sampler_matches_exact_enumeration(self):
        init, edge, frame, _, _, paths, log_q, _ = binary_chain()
        draws = 12_000
        samples = sample_hmm_paths(
            init.detach().expand(draws, -1), frame.detach().expand(draws, -1, -1),
            edge.detach().expand(draws, -1, -1, -1), torch.full((draws,), 4),
            generator=torch.Generator().manual_seed(414),
        )
        ids = (samples * torch.tensor([8, 4, 2, 1])).sum(-1)
        empirical = torch.bincount(ids, minlength=len(paths)).double() / draws
        self.assertLess(float((empirical - log_q.exp().detach()).abs().max()), 0.02)
        sampled_log_q = hmm_path_log_prob(
            init.detach(), frame.detach(), edge.detach(), torch.tensor([4]), samples[:1]
        )
        self.assertTrue(torch.isfinite(sampled_log_q).all())

    def test_score_function_gradient_matches_exact_expectation(self):
        init, edge, frame, _, _, paths, log_q, _ = binary_chain()
        rewards = torch.tensor([
            sum((1.0 if state else -0.35) * (t + 1) for t, state in enumerate(path.tolist()))
            for path in paths
        ], dtype=torch.double)
        exact_objective = -(log_q.exp() * rewards).sum()
        exact_grad = torch.autograd.grad(exact_objective, (init, edge), retain_graph=True)
        n = 50_000
        sampled_ids = torch.multinomial(
            log_q.exp().detach(), n, replacement=True, generator=torch.Generator().manual_seed(142)
        )
        sampled_paths = paths[sampled_ids]
        sampled_score = init[0, sampled_paths[:, 0]] + frame[0, 0, sampled_paths[:, 0]]
        for t in range(1, 4):
            sampled_score = sampled_score + edge[0, t-1, sampled_paths[:, t-1], sampled_paths[:, t]]
            sampled_score = sampled_score + frame[0, t, sampled_paths[:, t]]
        all_scores = init[0, paths[:, 0]] + frame[0, 0, paths[:, 0]]
        for t in range(1, 4):
            all_scores = all_scores + edge[0, t-1, paths[:, t-1], paths[:, t]]
            all_scores = all_scores + frame[0, t, paths[:, t]]
        sampled_log_q = sampled_score - torch.logsumexp(all_scores, dim=0)
        estimate = score_function_loss(
            rewards[sampled_ids], sampled_log_q, torch.zeros(n, dtype=torch.double)
        )
        estimated_grad = torch.autograd.grad(estimate, (init, edge))
        for actual, expected in zip(estimated_grad, exact_grad):
            self.assertLess(float((actual - expected).abs().max()), 0.045)

    def test_text_prior_and_geometry_helpers(self):
        prior = TextContactPrior(text_dim=3, hidden_dim=5).double()
        text = torch.randn(2, 3, dtype=torch.double)
        lengths = torch.tensor([1, 4])
        paths = prior.sample(text, lengths, generator=torch.Generator().manual_seed(9))
        self.assertEqual(tuple(paths.shape), (2, 2, 4))
        self.assertTrue(torch.equal(paths[0, :, 1:], paths[0, :, :1].expand(-1, 3)))
        self.assertTrue(torch.isfinite(prior.log_prob(paths, text, lengths)).all())

        mean = torch.zeros(1, 6, dtype=torch.double, requires_grad=True)
        covariance = (torch.eye(6, dtype=torch.double) * 0.2).unsqueeze(0).requires_grad_()
        sample = sample_geometry_prior(mean, covariance, torch.Generator().manual_seed(13))
        mean_grad, covariance_grad = torch.autograd.grad(sample.square().sum(), (mean, covariance))
        self.assertTrue(torch.isfinite(mean_grad).all())
        self.assertTrue(torch.isfinite(covariance_grad).all())


if __name__ == "__main__":
    unittest.main()
