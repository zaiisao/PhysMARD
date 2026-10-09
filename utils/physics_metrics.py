"""Foot penetration diagnostics; these measurements do not affect training losses."""

from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def fixed_rng(seed):
    """Make diagnostics repeatable without advancing training random streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@torch.no_grad()
def get_physics_metrics(joints, valid_frames, tolerance=0.001):
    """Measure raw foot heights against Y=0, without calibration bias correction."""
    foot_y = joints.detach()[..., [7, 10, 8, 11], 1]
    heights = foot_y[valid_frames[..., None].expand_as(foot_y)]
    depth = torch.relu(-heights)
    return {
        "foot_pen_mean": depth.mean(),
        "foot_pen_max": depth.max(),
        "foot_pen_rate": (heights < -tolerance).float().mean(),
    }


def make_generation_panel(dataset, seed, count=32):
    """Capture fixed captions/lengths from a seeded validation subset."""
    with fixed_rng(seed):
        indices = torch.randperm(len(dataset))[:count].tolist()
        panel = []
        for index in indices:
            caption, _, frames = dataset[index]
            panel.append({
                "dataset_index": index,
                "caption": caption,
                "frames": int(frames // 4 * 4),
            })
    return panel


@torch.no_grad()
def evaluate_generation_penetration(model, ae, panel, mean, std, seed):
    """Evaluate a fixed EMA generation panel using standard HumanML3D sampling."""
    from utils.motion_process import recover_from_ric

    device = next(model.parameters()).device
    model_training, ae_training = model.training, ae.training
    try:
        model.eval()
        ae.eval()
        with fixed_rng(seed):
            joints_batches = []
            # Fixed microbatches bound decoding memory and preserve sampling order.
            for start in range(0, len(panel), 8):
                batch = panel[start:start + 8]
                lengths = torch.tensor([row["frames"] for row in batch], device=device)
                latent = model.generate(
                    [row["caption"] for row in batch], lengths // 4,
                    timesteps=18, cond_scale=4.5, temperature=1.0,
                )
                motion = ae.decode(latent).float()
                physical = motion * std + mean
                joints = recover_from_ric(physical, 22)
                valid = torch.arange(motion.shape[1], device=device)[None, :] < lengths[:, None]
                # Flatten valid observations to combine batches without padding bias.
                joints_batches.append(joints[valid])
            joints = torch.cat(joints_batches).unsqueeze(0)
            valid = torch.ones(joints.shape[:2], dtype=torch.bool, device=device)
            return {name: value.item() for name, value in get_physics_metrics(joints, valid).items()}
    finally:
        model.train(model_training)
        ae.train(ae_training)
