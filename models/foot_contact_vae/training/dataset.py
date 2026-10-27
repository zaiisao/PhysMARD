"""HumanML3D clip selection, sequence batching, and encoder-statistic helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..source.geometry import FOOT_JOINTS
from ..source.model import FPS, foot_frame

JOINT_COUNT = 22
COORDINATE_COUNT = 3


def select_train_validation_clips(data_root, train_count, validation_count, seed):
    data_root = Path(data_root)
    train_file = data_root / "train.txt"
    lines = train_file.read_text().splitlines()
    names = [line.strip() for line in lines if line.strip()]

    np.random.default_rng(seed).shuffle(names)

    train_names, validation_names = [], []
    for clip in names:
        joints = np.load(data_root / "new_joints" / f"{clip}.npy", mmap_mode="r")
        if joints.ndim != 3 or joints.shape[1:] != (JOINT_COUNT, COORDINATE_COUNT):
            continue

        if len(train_names) < train_count:
            train_names.append(clip)
        elif len(validation_names) < validation_count:
            validation_names.append(clip)
        if len(train_names) == train_count and len(validation_names) == validation_count:
            break

    if len(train_names) != train_count or len(validation_names) != validation_count:
        raise RuntimeError("Not enough valid clips for the requested split sizes")

    return train_names, validation_names


def load_foot_sequences(ids, data_root):
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


def encoder_input_statistics(rows):
    """Return mean and standard deviation for the encoder input features."""
    batch = collate(rows)
    ankle, toe = batch["ankle"], batch["toe"]
    _, _, _, ankle_to_toe_angle = foot_frame(ankle, toe)

    ankle_velocity = FPS * torch.linalg.vector_norm(
        ankle[:, 1:] - ankle[:, :-1], dim=-1, keepdim=True
    )
    toe_velocity = FPS * torch.linalg.vector_norm(
        toe[:, 1:] - toe[:, :-1], dim=-1, keepdim=True
    )

    ankle_speed = torch.cat((ankle_velocity, ankle_velocity[:, -1:]), dim=1)
    toe_speed = torch.cat((toe_velocity, toe_velocity[:, -1:]), dim=1)

    features = torch.cat((
        ankle[..., 1:2], toe[..., 1:2], ankle_to_toe_angle, ankle_speed, toe_speed
    ), dim=-1)
    valid = torch.arange(features.shape[1])[None] < batch["lengths"][:, None]
    values = features[valid]
    mean, sd = values.mean(0), values.std(0)

    return mean, sd
