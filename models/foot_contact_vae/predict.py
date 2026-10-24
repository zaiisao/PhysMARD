"""CLI: infer foot-contact probabilities from a HumanML3D [T,22,3] .npy file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .inference import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_model, predict_contacts
from .source.model import FPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="HumanML3D joint array (.npy), shape [T,22,3]")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--geometry-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=95095)
    args = parser.parse_args()

    source, output = Path(args.input), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, args.device)
    joints = np.load(source).astype(np.float32)
    result = predict_contacts(joints, model, args.geometry_samples, args.seed)
    np.savez_compressed(output, **result)
    config = json.loads(DEFAULT_CONFIG.read_text())
    sidecar = {
        "input": str(source.resolve()),
        "output": str(output.resolve()),
        "input_shape": list(joints.shape),
        "fps": FPS,
        "floor_height_m": config["floor_m"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "geometry_samples": args.geometry_samples,
        "seed": args.seed,
        "device": str(next(model.parameters()).device),
        "states": result["state_names"].tolist(),
        "side_order": result["side_names"].tolist(),
        "motion_modified": False,
        "arrays": {
            "state_probability": "[side, frame, state] marginal probabilities",
        },
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps({"input": str(source), "output": str(output), "frames": len(joints)}))


if __name__ == "__main__":
    main()
