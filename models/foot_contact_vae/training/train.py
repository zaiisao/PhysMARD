"""Train the HumanML3D VAE and save a self-contained run directory.

Run from the repository root with:
    python -m models.foot_contact_vae.training.train --dataset-root /path/to/HumanML3D
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.foot_contact_vae.inference import elbo_mc
from models.foot_contact_vae.source.model import GEOMETRY_DIM, NUM_ENCODER_FEATURES, StructuredFootVAE
from .dataset import batches, clip_ids, collate, input_statistics, sequence_rows

TRAINING_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TRAINING_DIR.parent
DEFAULT_CONFIG = TRAINING_DIR / "config.json"
CONFIG = json.loads(DEFAULT_CONFIG.read_text())
RUN_DIR = None
CHECKPOINT_DIR = None
LOG_DIR = None

SCALE_INITIALS = {
    "raw_contact_sd": 0.01,
    "raw_air_sd": 0.08,
    "raw_flat_pitch_sd": 0.05,
    "raw_pitch_sd": 0.50,
    "raw_stick_speed_sd": 0.03,
    "raw_slide_speed_sd": 0.50,
    "raw_free_speed_sd": 1.0,
}


class ELBOBatchModule(nn.Module):
    def __init__(self, model, geometry_samples):
        super().__init__()
        self.model = model
        self.geometry_samples = geometry_samples

    def forward(self, ankle, toe, lengths, sides, eps_by_sequence):
        batch = {"ankle": ankle, "toe": toe, "lengths": lengths, "sides": sides}
        elbo, _ = elbo_mc(
            self.model, batch, self.geometry_samples,
            eps=eps_by_sequence.transpose(0, 1).contiguous(),
        )
        return elbo


def inverse_softplus(value):
    return float(np.log(np.expm1(value)))


def initialize_observation_parameters(model):
    with torch.no_grad():
        for name, value in SCALE_INITIALS.items():
            getattr(model.foot_contact_emission, name).fill_(inverse_softplus(value))
        model.foot_contact_emission.air_mean.fill_(0.10)
        model.foot_contact_emission.pitch_mean.fill_(-0.30)


def set_trainable_stage(model, stage):
    if stage not in {"fixed_scales", "scale_finetune"}:
        raise ValueError(stage)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name in {"prior_init_logits", "prior_transition_logits"} or name.startswith("geometry_posterior."):
            parameter.requires_grad_(True)
        if stage == "scale_finetune" and name in {
            "foot_contact_emission.air_mean", "foot_contact_emission.pitch_mean", "foot_contact_emission.raw_contact_sd",
            "foot_contact_emission.raw_air_sd", "foot_contact_emission.raw_flat_pitch_sd", "foot_contact_emission.raw_pitch_sd",
            "foot_contact_emission.raw_stick_speed_sd", "foot_contact_emission.raw_free_speed_sd",
        }:
            parameter.requires_grad_(True)


def state_to_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def log_scales(model):
    values = {key: float(torch.nn.functional.softplus(getattr(model.foot_contact_emission, name)).detach())
              for key, name in {
                  "contact_sd_m": "raw_contact_sd", "air_sd_m": "raw_air_sd",
                  "flat_pitch_sd_rad": "raw_flat_pitch_sd", "pitch_sd_rad": "raw_pitch_sd",
                  "stick_speed_sd_mps": "raw_stick_speed_sd", "slide_speed_sd_mps": "raw_slide_speed_sd",
                  "free_speed_sd_mps": "raw_free_speed_sd",
              }.items()}
    values["air_mean_m"] = float(model.foot_contact_emission.air_mean.detach())
    values["pitch_mean_rad"] = float(model.foot_contact_emission.pitch_mean.detach())
    return values


def train_stage(model, objective, device, train_rows, validation_batches, stage, epochs, seed):
    set_trainable_stage(model, stage)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=CONFIG["training"]["learning_rate"])
    rng = np.random.default_rng(seed)
    samples = CONFIG["training"]["geometry_samples"]
    best_value, best_state = -float("inf"), None
    history, start = [], time.time()
    log_path = LOG_DIR / f"{stage}.jsonl"
    with log_path.open("w") as handle:
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for rows in batches(train_rows, CONFIG["training"]["batch_sequences"], True, rng):
                batch = collate(rows, device=device)
                eps = torch.randn(len(rows), samples, GEOMETRY_DIM, device=device)
                loss = -objective(batch["ankle"], batch["toe"], batch["lengths"],
                                  batch["sides"], eps).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at {stage}, epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, CONFIG["training"]["gradient_clip_norm"])
                optimizer.step()
                losses.append(float(loss.detach()))

            model.eval()
            with torch.no_grad():
                validation = [float(objective(batch["ankle"], batch["toe"], batch["lengths"],
                                              batch["sides"], eps).mean())
                              for batch, eps in validation_batches]
            row = {"stage": stage, "epoch": epoch, "train_elbo_mean": -float(np.mean(losses)),
                   "validation_elbo_mean": float(np.mean(validation)),
                   "seconds_elapsed": time.time() - start, "scales": log_scales(model)}
            history.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(json.dumps(row), flush=True)
            if row["validation_elbo_mean"] > best_value:
                best_value = row["validation_elbo_mean"]
                best_state = state_to_cpu(model)
                torch.save(best_state, CHECKPOINT_DIR / f"best_{stage}.pt")

    if best_state is None:
        raise RuntimeError(f"No checkpoint produced for {stage}")
    model.load_state_dict(best_state)
    best_epoch = max(history, key=lambda item: item["validation_elbo_mean"])["epoch"]
    (LOG_DIR / f"{stage}_summary.json").write_text(json.dumps({
        "stage": stage, "epochs_requested": epochs, "best_epoch": best_epoch,
        "best_validation_elbo": best_value, "elapsed_seconds": time.time() - start,
        "scales": log_scales(model),
    }, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=CONFIG["dataset_root"])
    parser.add_argument("--device", default=CONFIG["device"])
    parser.add_argument("--run-dir", help="Output directory; defaults to a timestamped folder under package/runs")
    args = parser.parse_args(argv)

    global RUN_DIR, CHECKPOINT_DIR, LOG_DIR
    RUN_DIR = Path(args.run_dir) if args.run_dir else PACKAGE_DIR / "runs" / f"humanml3d_{datetime.now():%Y%m%d_%H%M%S}"
    CHECKPOINT_DIR, LOG_DIR = RUN_DIR / "checkpoints", RUN_DIR / "logs"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=False)
    LOG_DIR.mkdir(parents=True, exist_ok=False)

    seed = CONFIG["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing a CPU fallback")

    splits, split_seeds = CONFIG["split_sizes_clips"], CONFIG["split_seeds"]
    train_ids = clip_ids(args.dataset_root, "train", splits["train"], split_seeds["train"])
    validation_ids = clip_ids(args.dataset_root, "train", splits["validation"], split_seeds["validation"], train_ids)
    test_ids = clip_ids(args.dataset_root, "test", splits["test"], split_seeds["test"])
    train_rows = sequence_rows(train_ids, args.dataset_root)
    validation_rows = sequence_rows(validation_ids, args.dataset_root)

    probe = StructuredFootVAE(torch.zeros(NUM_ENCODER_FEATURES), torch.ones(NUM_ENCODER_FEATURES), floor_height=CONFIG["floor_m"])
    input_mean, input_sd = input_statistics(train_rows, probe, CONFIG["training"]["input_statistics_rows"])
    model = StructuredFootVAE(input_mean, input_sd, floor_height=CONFIG["floor_m"],
                             hidden=CONFIG["model"]["hidden"]).to(device)
    initialize_observation_parameters(model)
    objective = ELBOBatchModule(model, CONFIG["training"]["geometry_samples"]).to(device)

    validation_rng = torch.Generator(device=device).manual_seed(180731)
    validation_batches = []
    for rows in batches(validation_rows, 32, False, np.random.default_rng(0)):
        batch = collate(rows, device=device)
        eps = torch.randn(len(rows), CONFIG["training"]["validation_geometry_samples"], GEOMETRY_DIM,
                          generator=validation_rng, device=device)
        validation_batches.append((batch, eps))

    manifest = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "train_clips": train_ids, "validation_clips": validation_ids, "test_clips": test_ids,
        "train_sequences": len(train_rows), "validation_sequences": len(validation_rows),
        "test_sequences": 2 * len(test_ids), "input_mean": input_mean.tolist(), "input_sd": input_sd.tolist(),
        "config": CONFIG,
    }
    (RUN_DIR / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    zero_eps = torch.zeros(1, 2, GEOMETRY_DIM, device=device)
    probe_batch = collate(train_rows[:2], device=device)
    with torch.no_grad():
        production = model(probe_batch["ankle"], probe_batch["toe"], probe_batch["lengths"],
                           probe_batch["sides"], sample=False)
        adapted, _ = elbo_mc(model, probe_batch, 1, eps=zero_eps)
        api_error = float((production - adapted).abs().max())
    if not np.isfinite(api_error) or api_error > 1e-4:
        raise RuntimeError(f"Model and Monte Carlo ELBO paths disagree: max error {api_error}")
    (RUN_DIR / "preflight.json").write_text(json.dumps({"forward_elbo_max_abs_error": api_error}, indent=2) + "\n")

    print(f"Training on {device}; output: {RUN_DIR}", flush=True)
    train_stage(model, objective, device, train_rows, validation_batches, "fixed_scales",
                CONFIG["scale_schedule"]["fixed_scales_epochs"], seed)
    torch.save(state_to_cpu(model), CHECKPOINT_DIR / "final_fixed_scales.pt")
    train_stage(model, objective, device, train_rows, validation_batches, "scale_finetune",
                CONFIG["scale_schedule"]["scale_finetune_epochs"], seed + 1)
    final_state = state_to_cpu(model)
    torch.save(final_state, CHECKPOINT_DIR / "best.pt")
    torch.save(final_state, CHECKPOINT_DIR / "final.pt")
    metadata = {
        "checkpoint": str(CHECKPOINT_DIR / "best.pt"), "run_dir": str(RUN_DIR),
        "best_validation_elbo": json.loads((LOG_DIR / "scale_finetune_summary.json").read_text())["best_validation_elbo"],
        "floor_m": CONFIG["floor_m"], "device": str(device),
    }
    (CHECKPOINT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
