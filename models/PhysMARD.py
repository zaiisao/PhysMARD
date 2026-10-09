"""Physics-aware MARDM training with sequence-summed DDPM VLB accounting."""

import torch

from models.MARDM import MARDM
from utils.motion_process import recover_from_ric
from utils.physics_metrics import get_physics_metrics


class PhysMARD(MARDM):
    """Reuse MARDM architecture and inference; extend its training objective."""

    def forward_loss(self, latents, y, m_lens, *, decoder=None,
                     reference_motion=None, mean=None, std=None,
                     foot_height_bias=None, foot_height_sigma=None):
        """Return the combined training loss and detached logging metrics."""
        result = super().forward_loss(
            latents, y, m_lens,
            shared_timesteps=True, return_details=True,
        )
        data_loss = result["loss"]
        sequence_t = result["sequence_t"]
        clean_sequence = result["clean_sequence"]
        sequence_mask = result["sequence_mask"]
        b, l, d = clean_sequence.shape

        predicted_tokens = result["predicted_tokens"]
        indices = sequence_mask.reshape(-1).nonzero(as_tuple=True)[0]
        predicted_sequence = clean_sequence.reshape(b * l, d).index_copy(
            0,
            indices,
            predicted_tokens[:indices.numel()],
        )

        # Decoder expects [batch, latent_channels, sequence_length].
        predicted_latents = predicted_sequence.reshape(b, l, d).permute(0, 2, 1)
        predicted_motion = decoder(predicted_latents)

        physics_nll, physics_metrics = self.physics_nll(
            predicted_motion, reference_motion,
            frame_lengths=m_lens * 4,
            mean=mean,
            std=std,
            foot_height_bias=foot_height_bias,
            foot_height_sigma=foot_height_sigma,
            sequence_t=sequence_t,
        )

        loss = data_loss + physics_nll
        physics_metrics.update(data_loss=data_loss.detach(), physics_nll=physics_nll.detach())

        return loss, physics_metrics

    # Added by JA: virtual-observation likelihood on predicted motion.
    def physics_nll(
        self,
        predicted_motion,
        reference_motion,
        frame_lengths,
        mean,
        std,
        foot_height_bias=None,
        foot_height_sigma=None,
        sequence_t=None,
    ):
        """Prepare physical motion once and sum the constraint likelihoods."""
        # reference_motion is reserved for future contact supervision.
        # The observation likelihood has no additional timestep weighting.
        if predicted_motion.ndim != 3 or predicted_motion.shape[-1] != 67:
            raise ValueError("Expected HumanML3D motion with shape [B, F, 67]")

        # Fixed normalization and likelihood parameters are validated during calibration.
        b, frames, _ = predicted_motion.shape
        device = predicted_motion.device
        # Float32 recovery and log-CDF evaluation preserve the decoder gradient.
        motion = predicted_motion.float()

        mean = torch.as_tensor(mean, device=device, dtype=motion.dtype)
        std = torch.as_tensor(std, device=device, dtype=motion.dtype)
        lengths = frame_lengths.to(device=device, dtype=torch.long)

        if lengths.shape != (b,) or not ((lengths > 0) & (lengths <= frames)).all():
            raise ValueError("Frame lengths must have shape [B] and lie in [1, F]")

        motion_physical = motion * std[None, None, :] + mean[None, None, :]

        # JA: With joints_num=22, recover_from_ric reads 3 * (22 - 1) position
        # features for the non-root joints, undoes root yaw, adds root XZ
        # translation, and prepends the recovered root position. This gives
        # XYZ positions for all HumanML3D joints with shape [B, F, 22, 3].
        joints = recover_from_ric(motion_physical, 22)
        valid_frames = torch.arange(frames, device=device)[None, :] < lengths[:, None]

        # p(o_ground=1 | x_hat) = Phi((foot_y - bias) / sigma).
        # L_ground = -mean_batch(sum_valid_frames_and_feet(log p)).
        # ELBO contribution: E_q[log p(O_ground | x)]; here evaluated at x_hat.
        ground_penetration_nll = self.ground_penetration_nll(
            joints, valid_frames, foot_height_bias, foot_height_sigma, sequence_t,
        )

        # Contact likelihood p(O_contact | x_hat): formulation pending.
        # L_contact = -mean_batch(log p(O_contact | x_hat)).
        # ELBO contribution: E_q[log p(O_contact | x)]; disabled for now.
        contact_consistency_nll = 0.0

        # Balance likelihood p(O_balance | x_hat): formulation pending.
        # L_balance = -mean_batch(log p(O_balance | x_hat)).
        # ELBO contribution: E_q[log p(O_balance | x)]; disabled for now.
        balance_nll = 0.0

        # Energy likelihood p(O_energy | x_hat): formulation pending.
        # L_energy = -mean_batch(log p(O_energy | x_hat)).
        # ELBO contribution: E_q[log p(O_energy | x)]; disabled for now.
        mechanical_energy_consistency_nll = 0.0

        physics_nll = (
            ground_penetration_nll
            + contact_consistency_nll
            + balance_nll
            + mechanical_energy_consistency_nll
        )
        physics_metrics = get_physics_metrics(joints, valid_frames)

        return physics_nll, physics_metrics

    def ground_penetration_nll(
        self, joints, valid_frames, foot_height_bias, foot_height_sigma, sequence_t,
    ):
        """Sum calibrated foot-ground NLLs, then average over the batch."""
        foot_y = joints[..., [7, 10, 8, 11], 1]
        valid = valid_frames[..., None]

        # Give padding a benign log-CDF input before excluding its likelihood.
        foot_y = torch.where(
            valid, foot_y, torch.full_like(foot_y, foot_height_bias),
        )
        alpha_bar = torch.as_tensor(
            self.DiffMLPs.train_diffusion.alphas_cumprod,
            device=joints.device,
            dtype=torch.float64,
        )[sequence_t.to(device=joints.device, dtype=torch.long)]
        sigma_phys_t = (
            torch.as_tensor(foot_height_sigma, device=joints.device, dtype=torch.float64)
            / alpha_bar.sqrt()
        )
        # Float64 avoids log-CDF gradient cancellation for extreme early predictions.
        standardized_height = (
            foot_y.double() - foot_height_bias
        ) / sigma_phys_t[:, None, None]
        point_nll = -torch.special.log_ndtr(standardized_height)
        sequence_nll = torch.where(valid, point_nll, torch.zeros_like(point_nll)).sum((1, 2))
        return sequence_nll.mean()


def physmard_ddpm_xl(**kwargs):
    return PhysMARD(
        latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16,
        dropout=0.2, clip_dim=512, diffmlps_model="DDPM-XL",
        diffmlps_batch_mul=4, cond_drop_prob=0.1, **kwargs,
    )


