"""HumanML3D clip selection, sequence batching, and encoder-statistic helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..source.geometry import FOOT_JOINTS


def clip_ids(data_root, split: str, count: int, seed: int, exclude=()):
    data_root = Path(data_root)
    excluded = set(exclude)
    ids = [line.strip() for line in (data_root / f"{split}.txt").read_text().splitlines()
           if line.strip() and line.strip() not in excluded and not line.strip().startswith("M")]
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    selected = []
    for clip in ids:
        joints = np.load(data_root / "new_joints" / f"{clip}.npy", mmap_mode="r")
        if joints.ndim == 3 and joints.shape[1:] == (22, 3) and 40 <= len(joints) <= 200:
            selected.append(clip)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Only selected {len(selected)} of {count} {split} clips")


def sequence_rows(ids, data_root):
    data_root = Path(data_root)
    rows = []
    for clip in ids:
        joints = np.load(data_root / "new_joints" / f"{clip}.npy").astype(np.float32)
        for side_id, (side, (ankle, toe)) in enumerate(FOOT_JOINTS.items()):
            rows.append({"clip": clip, "side": side, "side_id": side_id,
                         "ankle": joints[:, ankle], "toe": joints[:, toe]})
    return rows


def collate(rows, device="cpu"):
    frames = max(len(row["ankle"]) for row in rows)

    def padded(key):
        values = []
        for row in rows:
            value = row[key]
            if len(value) < frames:
                value = np.concatenate((value, np.repeat(value[-1:], frames - len(value), axis=0)))
            values.append(value)
        return torch.as_tensor(np.stack(values), dtype=torch.float32, device=device)

    return {
        "ankle": padded("ankle"), "toe": padded("toe"),
        "lengths": torch.tensor([len(row["ankle"]) for row in rows], dtype=torch.long, device=device),
        "sides": torch.tensor([row["side_id"] for row in rows], dtype=torch.long, device=device),
        "clips": [row["clip"] for row in rows], "side_names": [row["side"] for row in rows],
    }


def batches(rows, size, shuffle, rng):
    ordered = sorted(range(len(rows)), key=lambda index: len(rows[index]["ankle"]))
    chunks = [ordered[start:start + size] for start in range(0, len(ordered), size)]
    if shuffle:
        rng.shuffle(chunks)
    return [[rows[index] for index in chunk] for chunk in chunks]


def input_statistics(rows, model, count=400):
    batch = collate(rows[:count])
    features = torch.cat(model.encoder_inputs(batch["ankle"], batch["toe"]), dim=-1)
    valid = torch.arange(features.shape[1])[None] < batch["lengths"][:, None]
    values = features[valid]
    mean, sd = values.mean(0), values.std(0)
    if not torch.isfinite(mean).all() or not torch.isfinite(sd).all() or not (sd > 0).all():
        raise RuntimeError("Invalid encoder input statistics")
    return mean, sd
