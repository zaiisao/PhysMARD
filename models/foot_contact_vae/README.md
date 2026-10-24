# Frozen structured foot-contact VAE source copy

This directory contains a partial source copy of the structured foot-contact VAE for reference. The experiment-specific `RealVAE` wrapper is not included. Nothing here is wired into `PhysMARD` or the MARDM training path.

## Files

- `source/model.py` — structured geometry and seven-state foot-contact model.
- `source/geometry.py` and `source/support_geometry.json` — support-point geometry and the cached SMPL prior.

The source was copied from `/home/sogang/latent_foot_vae_fixed_floor/geometry_marginalized/`.

Local run outputs and checkpoints under `source/runs_real/` are ignored and are not included in Git. This source copy is not wired into `PhysMARD` or the MARDM training path.
