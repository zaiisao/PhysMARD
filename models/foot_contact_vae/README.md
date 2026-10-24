# Frozen structured foot-contact VAE source copy

This directory contains a read-through copy of the frozen VAE used by the visual-contact diagnostics. It is not wired into `PhysMARD` or the MARDM training path.

## Files

- `source/vae_real.py` — real-motion VAE encoder configuration.
- `source/model.py` — structured geometry and seven-state foot-contact model.
- `source/geometry.py` and `source/support_geometry.json` — support-point geometry and the cached SMPL prior.

The source was copied from `/home/sogang/latent_foot_vae_fixed_floor/geometry_marginalized/`.

Local run outputs and checkpoints under `source/runs_real/` are ignored and are not included in Git. This source copy is not wired into `PhysMARD` or the MARDM training path.
