# Structured foot-contact VAE

This package predicts foot-contact states from a HumanML3D joint sequence. It does **not** change the input motion.

## Input → output

Input is a NumPy array with shape `[T, 22, 3]`, in HumanML3D joint order and world coordinates in meters, sampled at 20 fps. The floor is fixed at `y = -0.0109 m`. The VAE runs both feet independently.

Run from the repository root:

```bash
python -m models.foot_contact_vae.predict \
  --input /disk4/jaehoon/HumanML3D/new_joints/005095.npy \
  --output /tmp/005095_contacts.npz \
  --device cuda:2
```

The `.npz` contains one prediction array, `state_probability[side, frame, state]`, plus the state and side names. The seven-state probabilities include air, contact, sticking, and sliding information; derive those group probabilities or the most likely state from this array as needed. The JSON sidecar records the input, checkpoint, settings, and array shape. Detailed geometry and support-point diagnostics remain in the analysis outputs.

Python entry point:

```python
import numpy as np
from models.foot_contact_vae import load_model, predict_contacts

model = load_model(device="cuda:2")
joints = np.load("/path/to/clip.npy")  # [T, 22, 3]
prediction = predict_contacts(joints, model)
print(prediction["state_probability"].shape)  # [2, T, 7]
```

## Model and checkpoint

- `source/model.py` contains the seven-state model and HumanML3D input normalization. It uses five foot features and an unconditioned state prior; the existing checkpoint was converted once to remove its zero context channels.
- `source/geometry.py` and `source/support_geometry.json` provide the fixed left/right heel, pad, and tip geometry prior. Model loading reads the cached JSON and does not require the SMPL asset.
- `inference.py` loads the selected checkpoint and produces the seven-state probability output.
- `checkpoints/humanml3d_best.pt` is the local checkpoint used by the joint-contact experiments. It is ignored by Git; its training split, input statistics, source commit, and checksum are recorded in `training/trained_checkpoint_manifest.json` and `checkpoints/humanml3d_best.json`.

The checkpoint file is local and is not included in Git. The older, incompatible local run files are archived under `analysis/joint_contact_physics/legacy_vae_runs/`; they are not used by this package.

## Training

The canonical training entry point is `training/train.py`; its HumanML3D loader is `training/dataset.py`, and its settings are in `training/config.json`. It uses the train split for fitting and validation, keeps the test split separate, and saves each run under `models/foot_contact_vae/runs/` without overwriting the packaged checkpoint.

```bash
python -m models.foot_contact_vae.training.train \
  --dataset-root /disk4/jaehoon/HumanML3D \
  --device cuda:2
```

Use `--run-dir` to choose an explicit output directory. The run contains stage checkpoints, logs, a preflight record, and the exact split/input-statistics manifest.

## Experiment boundary

Vertical correction, latent motion refinement, virtual zero-velocity objectives, clip screening, and animation rendering remain under `analysis/joint_contact_physics/`. Those experiments call the VAE; they are not part of this inference package.
