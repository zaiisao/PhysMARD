"""Penetration diagnostic masking, repeatability, and gradient isolation."""

import random
import unittest

import numpy as np
import torch

from models.PhysMARD import PhysMARD
from utils.physics_metrics import (
    get_physics_metrics, make_generation_panel, evaluate_generation_penetration,
)


class FakeDataset:
    def __len__(self):
        return 40

    def __getitem__(self, index):
        return f'{index}-{random.randrange(100)}', None, 8 + 4 * np.random.randint(2)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def generate(self, captions, lengths, **kwargs):
        result = torch.zeros(len(captions), int(lengths.max()) * 4, 67)
        for joint in [7, 10, 8, 11]:
            result[..., 4 + (joint - 1) * 3 + 1] = -0.01 * torch.rand(())
        return result


class FakeAE(torch.nn.Module):
    def decode(self, latent):
        return latent


class PhysicsMetricsTests(unittest.TestCase):
    def test_masking_and_detachment(self):
        joints = torch.zeros(1, 3, 22, 3, requires_grad=True)
        with torch.no_grad():
            joints[0, 0, [7, 10, 8, 11], 1] = -0.002
            joints[0, 1, [7, 10, 8, 11], 1] = 0.01
            joints[0, 2, :, 1] = -100
        metrics = get_physics_metrics(joints, torch.tensor([[True, True, False]]))
        self.assertAlmostEqual(metrics['foot_pen_mean'].item(), 0.001)
        self.assertAlmostEqual(metrics['foot_pen_max'].item(), 0.002)
        self.assertEqual(metrics['foot_pen_rate'].item(), 0.5)
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))

    def test_panel_and_generation_restore_rng_and_modes(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        panel = make_generation_panel(FakeDataset(), 17)
        self.assertEqual(panel, make_generation_panel(FakeDataset(), 17))
        self.assertEqual(len(panel), 32)
        model, ae = FakeModel().train(), FakeAE().train()
        kwargs = dict(mean=torch.zeros(67), std=torch.ones(67), seed=17)
        first = evaluate_generation_penetration(model, ae, panel, **kwargs)
        self.assertEqual(first, evaluate_generation_penetration(model, ae, panel, **kwargs))
        self.assertTrue(model.training and ae.training)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_metrics_preserve_loss_and_gradient(self):
        model = PhysMARD.__new__(PhysMARD)
        motion = torch.zeros(1, 3, 67, requires_grad=True)
        kwargs = dict(reference_motion=None, frame_lengths=torch.tensor([2]),
                      mean=torch.zeros(67), std=torch.ones(67),
                      foot_height_bias=0., foot_height_sigma=.01)
        from utils.motion_process import recover_from_ric
        joints = recover_from_ric(motion, 22)
        plain = model.ground_penetration_nll(
            joints, torch.tensor([[True, True, False]]), 0., .01,
        )
        logged, metrics = model.physics_nll(motion, **kwargs)
        torch.testing.assert_close(plain, logged)
        torch.testing.assert_close(torch.autograd.grad(plain, motion)[0],
                                   torch.autograd.grad(logged, motion)[0])
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))


if __name__ == '__main__':
    unittest.main()
