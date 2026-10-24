# Frozen structured foot-contact VAE source copy

This directory contains a read-through copy of the frozen VAE used by the visual-contact diagnostics. It is not wired into `PhysMARD` or the MARDM training path.

## Files

- `source/vae_real.py` — real-motion VAE encoder and training/evaluation helpers.
- `source/model.py` — structured geometry and seven-state foot-contact model.
- `source/geometry.py` and `source/support_geometry.json` — support-point geometry and the cached SMPL prior.
- `source/runs_real/final.json` and `final.pt` — checkpoint metadata and frozen weights used by the diagnostics.

The source was copied from `/home/sogang/latent_foot_vae_fixed_floor/geometry_marginalized/`. The checkpoint SHA-256 is `37e8a924e1634f9166110112d7852250c357f6c7c52913fc9450185b53b4e29d`.

This is a source-and-checkpoint copy for inspection; imports, training data paths, and training entry points remain as in the standalone source. No PhysMARD behavior or training objective was changed by this copy.
