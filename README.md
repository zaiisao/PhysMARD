# PhysMARD

PhysMARD is a research fork of [MARDM](https://github.com/neu-vi/MARDM) exploring physics-informed training for text-driven human motion generation.

A pretrained autoencoder maps motion into latent tokens. A masked transformer supplies context to a DDPM token denoiser. Predicted clean tokens are reconstructed into a sequence and decoded before physical residuals are evaluated.

## Current status

- DDPM training with noise-prediction MSE or an optional continuous-latent variational objective (`--use_kl`).
- A frozen autoencoder with differentiable decoding for physics supervision.
- Shared diffusion timesteps within each sequence when physics is enabled.
- A `physics_nll()` hook inside `MARDM.forward_loss()`.

**Physics training is not implemented yet.** `--physics` currently raises `NotImplementedError`. Physical residuals, observation covariances, and consistent data/physics ELBO scaling remain to be implemented. The current scaffold does not establish an exact combined ELBO or guarantee physically valid generated motion.

The intended physics formulation uses auxiliary observations of decoded clean predictions, informed by [Physics-Informed Diffusion Models](https://arxiv.org/abs/2403.14404). Generation currently uses the standard sampler without physics correction.

## Setup

```bash
conda env create -f environment.yml
conda activate MARDM
```

Prepare HumanML3D or KIT-ML using the [HumanML3D dataset instructions](https://github.com/EricGuo5513/HumanML3D). Place dataset files under `datasets/HumanML3D/` or `datasets/KIT-ML/`, including `new_joint_vecs/`, `texts/`, split files, and representation-compatible `Mean.npy` and `Std.npy`.

This implementation uses 67 motion features for HumanML3D and 64 for KIT-ML. Normalization statistics must match that representation.

Pretrained autoencoders, baseline models, length estimators, and evaluators are available through the [upstream model collection](https://huggingface.co/collections/cr8br0ze/mardm). Keep the extracted checkpoint layout:

```text
checkpoints/
  t2m/
    AE/model/latest.tar
    <experiment>/model/latest.tar
    length_estimator/model/finest.tar
  kit/
    AE/model/net_best_fid.tar
    <experiment>/model/latest.tar
```

Full evaluation additionally requires the upstream evaluation checkpoints and GloVe files under `glove/`. Sampling requires dataset normalization statistics and the length-estimator checkpoint.

## Training

HumanML3D DDPM baseline:

```bash
python train_MARDM.py --name DDPM_MSE --model MARDM-DDPM-XL --dataset_name t2m --ae_name AE
```

Continuous-latent variational objective:

```bash
python train_MARDM.py --name DDPM_KL --model MARDM-DDPM-XL --dataset_name t2m --ae_name AE --use_kl
```

Add `--is_continue` to resume an experiment, preserving its model and objective settings. Add `--need_evaluation --eval_every 20` to run generation metrics every 20 epochs. Data-loss validation runs every epoch.

For KIT-ML, use `--dataset_name kit` and its matching autoencoder and statistics.

## Sampling and evaluation

Use an experiment name matching your checkpoint directory:

```bash
python sample.py --name DDPM_KL --model MARDM-DDPM-XL --text_prompt "A person walks forward."
python evaluation_MARDM.py --name DDPM_KL --model MARDM-DDPM-XL --dataset_name t2m
```

The current training work targets DDPM. The inherited SiT training path has not been adapted to the new loss interface.

## Tests

```bash
python -m unittest discover -s tests -p 'test_continuous_elbo.py'
```

These tests cover the continuous Gaussian likelihood, DDPM variational terms, denoiser gradients, and gradient propagation through a frozen decoder. They do not validate the unfinished physics objective.

## Attribution

Based on the minimalist implementation of **Rethinking Diffusion for Text-Driven Human Motion Generation: Redundant Representations, Evaluation, and Masked Autoregression** (CVPR 2025), by Zichong Meng, Yiming Xie, Xiaogang Peng, Zeyu Han, and Huaizu Jiang.

- [Original MARDM repository](https://github.com/neu-vi/MARDM)
- [MARDM paper](https://arxiv.org/abs/2411.16575)
- [PIDM paper](https://arxiv.org/abs/2403.14404)
- [PIDM implementation](https://github.com/jhbastek/PhysicsInformedDiffusionModels)

See [LICENSE.txt](LICENSE.txt) for the repository license.
