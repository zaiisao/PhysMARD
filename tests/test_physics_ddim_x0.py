"""Added by JA: PIDM sample-estimation (DDIM x0) checks for the DDPM denoiser."""

import unittest

import torch

from models.DiffMLPs import DiffMLPs_DDPM


def make_denoiser():
    torch.manual_seed(0)
    model = DiffMLPs_DDPM(
        target_channels=3, z_channels=4, depth=2, width=16,
        num_sampling_steps="50", use_kl=True,
    )
    # The output layer is zero-initialized; randomize it so eps depends on x, t and c.
    for parameter in model.net.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    return model


class DDIMx0Tests(unittest.TestCase):
    def setUp(self):
        self.model = make_denoiser()
        generator = torch.Generator().manual_seed(1)
        self.target = torch.randn(6, 3, generator=generator, dtype=torch.float32)
        self.z = torch.randn(6, 4, generator=generator, dtype=torch.float32)
        self.t = torch.tensor([0, 1, 5, 20, 48, 49])

    def record_timesteps(self):
        calls = []
        forward = self.model.net.forward

        def recording(x, t, c):
            calls.append(t.tolist())
            return forward(x, t, c)

        self.model.net.forward = recording
        return calls

    def test_forward_exposes_noisy_input_without_changing_data_loss(self):
        torch.manual_seed(7)
        loss, pred_xstart, noisy = self.model(target=self.target, z=self.z, t=self.t)
        state_after = torch.get_rng_state()
        torch.manual_seed(7)
        noise = torch.randn_like(self.target)
        torch.manual_seed(7)
        reference = self.model.train_diffusion.training_losses(
            self.model.net, self.target, self.t, dict(c=self.z),
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), state_after))
        self.assertTrue(torch.equal(loss, reference["loss"].mean()))
        self.assertTrue(torch.equal(pred_xstart, reference["pred_xstart"]))
        expected = self.model.train_diffusion.q_sample(self.target, self.t, noise=noise)
        self.assertTrue(torch.equal(noisy, expected))

    def test_zero_timestep_returns_direct_prediction(self):
        torch.manual_seed(7)
        _, pred_xstart, noisy = self.model(target=self.target, z=self.z, t=self.t)
        for steps in (0, 1, 3):
            x0 = self.model.ddim_x0(noisy, self.t, self.z, reduced_steps=steps)
            torch.testing.assert_close(x0[0], pred_xstart[0], rtol=1e-6, atol=1e-6)

    def test_reduced_steps_zero_matches_manual_x_t_to_x_1_to_x_0(self):
        diffusion = self.model.train_diffusion
        x_t = diffusion.q_sample(self.target, self.t, noise=torch.randn_like(self.target))
        eps = self.model.net(x_t, self.t, self.z)
        x0_hat = diffusion._predict_xstart_from_eps(x_t=x_t, t=self.t, eps=eps)
        alpha_bar_1 = torch.tensor(diffusion.alphas_cumprod[0], dtype=torch.float32)
        x_1 = alpha_bar_1.sqrt() * x0_hat + (1 - alpha_bar_1).sqrt() * eps
        zero = torch.zeros_like(self.t)
        expected = diffusion._predict_xstart_from_eps(
            x_t=x_1, t=zero, eps=self.model.net(x_1, zero, self.z),
        )
        expected[0] = x0_hat[0]  # t == 0 never steps.
        torch.testing.assert_close(
            self.model.ddim_x0(x_t, self.t, self.z, reduced_steps=0), expected,
            rtol=1e-5, atol=1e-6,
        )
        # The final call must see the stepped state, not the original x_t.
        literal = diffusion._predict_xstart_from_eps(
            x_t=x_t, t=zero, eps=self.model.net(x_t, zero, self.z),
        )
        self.assertFalse(torch.allclose(expected[1:], literal[1:]))

    def test_timestep_schedule_matches_pidm(self):
        x_t = torch.zeros(1, 3, dtype=torch.float32)
        z = self.z[:1]
        t = torch.tensor([49])
        for steps, expected in (
            (0, [49, 0]),
            (1, [49, 24, 0]),
            (3, [49, 36, 24, 12, 0]),
        ):
            calls = self.record_timesteps()
            self.model.ddim_x0(x_t, t, z, reduced_steps=steps)
            self.assertEqual([call[0] for call in calls], expected)

    def test_gradients_reach_denoiser_at_high_timesteps(self):
        x_t = self.model.train_diffusion.q_sample(self.target, self.t)
        for steps in (0, 1, 3):
            self.model.zero_grad()
            self.model.ddim_x0(x_t, self.t, self.z, reduced_steps=steps).square().sum().backward()
            grads = [p.grad for p in self.model.net.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum() for g in grads).item(), 0)


if __name__ == '__main__':
    unittest.main()
