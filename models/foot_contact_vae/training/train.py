"""Train the HumanML3D VAE and save a self-contained run directory.

Run from the repository root with:
    python -m models.foot_contact_vae.training.train --dataset-root /path/to/HumanML3D
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from models.foot_contact_vae.source.model import GEOMETRY_DIM, StructuredFootVAE
from .dataset import (
    collate,
    encoder_input_statistics,
    load_foot_sequences,
    select_train_validation_clips,
)

TRAINING_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TRAINING_DIR.parent
DEFAULT_CONFIG = TRAINING_DIR / "config.yaml"


def load_config():
    return yaml.safe_load(DEFAULT_CONFIG.read_text())


def write_training_manifest(run_dir, dataset_root, train_ids, validation_ids, train_rows, validation_rows,
                            input_mean, input_sd, config):
    manifest_path = run_dir / "training_manifest.json"

    manifest = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "train_clips": train_ids,
        "validation_clips": validation_ids,
        "train_sequences": len(train_rows),
        "validation_sequences": len(validation_rows),
        "input_mean": input_mean.tolist(),
        "input_sd": input_sd.tolist(),
        "config": config,
    }
    manifest_json = json.dumps(manifest, indent=2) + "\n"
    manifest_path.write_text(manifest_json)


def train_model(model, config, device, train_sequences, validation_batches, checkpoint_dir, log_dir):
    training = config["training"]
    seed = config["seed"]
    geometry_samples = training["geometry_samples"]
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])

    best_validation_elbo = -float("inf")
    log_path = log_dir / "training.jsonl"
    with log_path.open("w") as handle:
        for epoch in range(training["epochs"]):
            shuffled_sequences = train_sequences.copy()
            np.random.shuffle(shuffled_sequences)

            total_training_loss = 0.0
            training_batch_count = 0
            batch_size = training["batch_sequences"]

            for start in range(0, len(shuffled_sequences), batch_size):
                rows = shuffled_sequences[start:start + batch_size]

                batch = collate(rows, device=device)
                loss = -model(
                    batch["ankle"], batch["toe"], batch["lengths"], batch["sides"],
                    samples=geometry_samples,
                ).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_training_loss += float(loss.detach())
                training_batch_count += 1

            validation_generator = torch.Generator(device=device).manual_seed(seed)
            validation_elbos = []
            with torch.no_grad():
                for batch in validation_batches:
                    validation_elbo = model(
                        batch["ankle"], batch["toe"], batch["lengths"], batch["sides"],
                        samples=geometry_samples, generator=validation_generator,
                    ).mean()
                    validation_elbos.append(float(validation_elbo))

            metrics = {
                "epoch": epoch + 1,
                "train_elbo_mean": -total_training_loss / training_batch_count,
                "validation_elbo_mean": float(np.mean(validation_elbos)),
            }

            log_line = json.dumps(metrics)
            handle.write(log_line + "\n")
            handle.flush()

            print(log_line, flush=True)

            if metrics["validation_elbo_mean"] > best_validation_elbo:
                best_validation_elbo = metrics["validation_elbo_mean"]
                checkpoint = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                torch.save(checkpoint, checkpoint_dir / "best.pt")


def main(argv=None):
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=config["dataset_root"])
    parser.add_argument("--device", default=config["device"])
    parser.add_argument("--run-dir", help="Output directory; defaults to a timestamped folder under package/runs")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else PACKAGE_DIR / "runs" / f"humanml3d_{datetime.now():%Y%m%d_%H%M%S}"
    checkpoint_dir, log_dir = run_dir / "checkpoints", run_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device)

    train_ids, validation_ids = select_train_validation_clips(
        args.dataset_root,
        config["split_sizes_clips"]["train"],
        config["split_sizes_clips"]["validation"],
        seed,
    )

    train_rows = load_foot_sequences(train_ids, args.dataset_root)
    validation_rows = load_foot_sequences(validation_ids, args.dataset_root)

    input_mean, input_sd = encoder_input_statistics(train_rows)
    model = StructuredFootVAE(
        input_mean, input_sd,
        hidden=config["model"]["hidden"],
    ).to(device)

    validation_batches = []
    for start in range(0, len(validation_rows), 32):
        rows = validation_rows[start:start + 32]
        batch = collate(rows, device=device)
        validation_batches.append(batch)

    write_training_manifest(
        run_dir, args.dataset_root, train_ids, validation_ids,
        train_rows, validation_rows, input_mean, input_sd, config,
    )

    print(f"Training on {device}; output: {run_dir}", flush=True)
    train_model(model, config, device, train_rows, validation_batches, checkpoint_dir, log_dir)

    print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
