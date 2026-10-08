"""Added by JA: analytic and gradient checks for continuous-latent DDPM training."""

import math
import unittest

import torch

from diffusions.diffusion import create_diffusion
from diffusions.diffusion.diffusion_utils import continuous_gaussian_log_likelihood
from diffusions.diffusion.gaussian_diffusion import LossType
from models.AE import AE
from models.DiffMLPs import DiffMLPs_DDPM


class NoiseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, x, t, **kwargs):
        return self.scale * x


class ContinuousELBOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = NoiseModel()
        self.clean = torch.tensor([[2.5, -3.0, 0.1], [0.3, 1.8, -2.1]])
        self.noise = torch.randn_like(self.clean)
        self.diffusion = create_diffusion(
            "", noise_schedule="cosine", use_kl=True,
            diffusion_steps=8,
        )

    def test_density_matches_normal(self):
        means = torch.randn_like(self.clean)
        log_scales = torch.randn_like(self.clean)
        actual = continuous_gaussian_log_likelihood(
            self.clean, means=means, log_scales=log_scales
        )
        expected = torch.distributions.Normal(means, log_scales.exp()).log_prob(self.clean)
        torch.testing.assert_close(actual, expected)

    def test_endpoint_nll_and_variance(self):
        t = torch.zeros(2, dtype=torch.long)
        noisy = self.diffusion.q_sample(self.clean, t, self.noise)
        out = self.diffusion.p_mean_variance(self.model, noisy, t, clip_denoised=False)
        torch.testing.assert_close(out["variance"], out["log_variance"].exp())
        torch.testing.assert_close(out["variance"], torch.full_like(self.clean, float(self.diffusion.posterior_variance[1])))
        expected = -torch.distributions.Normal(out["mean"], math.sqrt(self.diffusion.posterior_variance[1])).log_prob(
            self.clean
        ).mean(1) / math.log(2)
        actual = self.diffusion._vb_terms_bpd(
            self.model, self.clean, noisy, t, clip_denoised=False
        )["output"]
        torch.testing.assert_close(actual, expected)

    def test_rescaled_estimator_matches_explicit_sum(self):
        unscaled, scaled = [], []
        for step in range(self.diffusion.num_timesteps):
            t = torch.full((2,), step, dtype=torch.long)
            noisy = self.diffusion.q_sample(self.clean, t, self.noise)
            term = self.diffusion._vb_terms_bpd(
                self.model, self.clean, noisy, t, clip_denoised=False
            )["output"]
            self.diffusion.loss_type = LossType.KL
            raw = self.diffusion.training_losses(
                self.model, self.clean, t, noise=self.noise
            )["loss"]
            torch.testing.assert_close(raw, term)
            self.diffusion.loss_type = LossType.RESCALED_KL
            estimate = self.diffusion.training_losses(
                self.model, self.clean, t, noise=self.noise
            )["loss"]
            torch.testing.assert_close(estimate, raw * self.diffusion.num_timesteps)
            unscaled.append(term)
            scaled.append(estimate)
        torch.testing.assert_close(torch.stack(scaled).mean(0), torch.stack(unscaled).sum(0))

    def test_transition_kl_matches_torch(self):
        t = torch.tensor([1, 6])
        noisy = self.diffusion.q_sample(self.clean, t, self.noise)
        true_mean, true_var, _ = self.diffusion.q_posterior_mean_variance(self.clean, noisy, t)
        out = self.diffusion.p_mean_variance(self.model, noisy, t, clip_denoised=False)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Normal(true_mean, true_var.sqrt()),
            torch.distributions.Normal(out["mean"], out["variance"].sqrt()),
        ).mean(1) / math.log(2)
        actual = self.diffusion._vb_terms_bpd(
            self.model, self.clean, noisy, t, clip_denoised=False
        )["output"]
        torch.testing.assert_close(actual, expected)

    def test_mse_unchanged_and_both_losses_have_gradients(self):
        t = torch.tensor([0, 6])
        for loss_type in (LossType.MSE, LossType.RESCALED_KL):
            self.diffusion.loss_type = loss_type
            loss = self.diffusion.training_losses(
                self.model, self.clean, t, noise=self.noise
            )["loss"]
            if loss_type == LossType.MSE:
                noisy = self.diffusion.q_sample(self.clean, t, self.noise)
                expected = ((self.noise - self.model(noisy, t)) ** 2).mean(1)
                torch.testing.assert_close(loss, expected)
            self.model.zero_grad()
            loss.mean().backward()
            self.assertTrue(torch.isfinite(self.model.scale.grad).all())
            self.assertGreater(self.model.scale.grad.abs().item(), 0)

    def test_complete_bound_contains_prior(self):
        result = self.diffusion.calc_bpd_loop(self.model, self.clean, clip_denoised=False)
        torch.testing.assert_close(result["total_bpd"], result["vb"].sum(1) + result["prior_bpd"])
        self.assertTrue(torch.isfinite(result["total_bpd"]).all())

    def test_wrapper_and_frozen_decoder(self):
        for use_kl, expected in ((False, LossType.MSE), (True, LossType.RESCALED_KL)):
            wrapper = DiffMLPs_DDPM(8, 4, 1, 16, "4", use_kl=use_kl)
            self.assertEqual(wrapper.train_diffusion.loss_type, expected)
            # Modified by JA: unpack the loss, clean prediction, and timesteps.
            loss, predicted_latents, timesteps = wrapper(torch.randn(3, 8), torch.randn(3, 4))
            self.assertEqual(predicted_latents.shape, (3, 8))
            self.assertEqual(timesteps.shape, (3,))
            self.assertEqual(timesteps.dtype, torch.long)
            self.assertTrue(((timesteps >= 0) & (timesteps < wrapper.train_diffusion.num_timesteps)).all())
            loss.backward()
            grads = [p.grad for p in wrapper.parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
            self.assertTrue(any(g.abs().sum() > 0 for g in grads))
        ae = AE(input_width=67, output_emb_width=8, width=16, depth=1).eval()
        ae.requires_grad_(False)
        latent = torch.randn(2, 8, 4, requires_grad=True)
        ae.decode(latent).square().mean().backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(latent.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in ae.parameters()))


if __name__ == "__main__":
    unittest.main()
