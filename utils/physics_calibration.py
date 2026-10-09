"""Frozen-AE foot-height calibration for HumanML3D."""

import random

import numpy as np
import torch

from utils.motion_process import recover_from_ric


@torch.no_grad()
def calibrate_foot_height_error(ae, calibration_loader, mean, std):
    """Fit fixed Gaussian bias/std to valid training foot-height errors."""
    device = next(ae.parameters()).device
    ae.eval()

    mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(std, dtype=torch.float32, device=device)
    foot_indices = [7, 10, 8, 11]  # HumanML3D only.

    if mean.shape != (67,) or std.shape != (67,):
        raise ValueError("Calibration requires HumanML3D's 67 features.")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Normalization statistics must be finite.")
    if not (std > 0).all():
        raise ValueError("Normalization standard deviations must be positive.")

    # Float64 accumulation reduces numerical error in the variance estimate.
    error_sum = torch.zeros((), dtype=torch.float64, device=device)
    error_squared_sum = torch.zeros_like(error_sum)
    count = 0

    for _, motion, frame_lengths in calibration_loader:
        motion = motion.to(device=device, dtype=torch.float32)
        frame_lengths = frame_lengths.to(device=device, dtype=torch.long)

        reconstructed = ae.decode(ae.encode(motion))
        frames = reconstructed.shape[1]

        if (frame_lengths <= 0).any() or (frame_lengths > frames).any():
            raise ValueError(
                "Valid frame lengths must fit the AE reconstruction; "
                "use crops divisible by four."
            )

        # Both tensors use the training normalization statistics.
        gt_physical = motion[:, :frames] * std + mean
        recon_physical = reconstructed * std + mean

        gt_y = recover_from_ric(gt_physical, 22)[..., foot_indices, 1]
        recon_y = recover_from_ric(recon_physical, 22)[..., foot_indices, 1]

        valid_frames = (
            torch.arange(frames, device=device)[None, :]
            < frame_lengths[:, None]
        )
        valid = valid_frames[..., None].expand_as(gt_y)

        # Include every valid foot observation, not just penetrating feet.
        errors = (recon_y - gt_y)[valid].double()
        if not torch.isfinite(errors).all():
            raise ValueError("Nonfinite foot-height reconstruction errors.")

        error_sum += errors.sum()
        error_squared_sum += errors.square().sum()
        count += errors.numel()

    if count == 0:
        raise ValueError("Calibration loader contained no valid observations.")

    bias = error_sum / count
    variance = error_squared_sum / count - bias.square()
    sigma = variance.clamp_min(0).sqrt()

    bias = bias.item()
    sigma = sigma.item()

    if not np.isfinite(bias):
        raise ValueError("Calibration produced an invalid foot-height bias.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Calibration produced an invalid uncertainty scale.")

    print(
        f"Foot-height calibration: observations={count}, "
        f"bias={bias:.8g}, sigma={sigma:.8g}"
    )
    return bias, sigma


def calibrate_foot_height_error_seeded(ae, calibration_loader, mean, std, seed):
    """Run repeatable calibration without advancing training RNG states."""
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    try:
        with torch.random.fork_rng():
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            return calibrate_foot_height_error(ae, calibration_loader, mean, std)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
