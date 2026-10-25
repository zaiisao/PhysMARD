"""Infer foot-contact probabilities from a HumanML3D joint array."""

import argparse
from pathlib import Path

import numpy as np

from .inference import (
    DEFAULT_CHECKPOINT,
    load_model,
    predict_contacts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="HumanML3D joint array (.npy), shape [T,22,3]",
    )
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", required=True, help="cpu, cuda, or cuda:N")
    parser.add_argument("--geometry-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=95095)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)

    # Ensure the output folder exists before writing predictions.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joints = np.load(input_path)

    model = load_model(args.device, checkpoint=checkpoint_path)
    predictions = predict_contacts(
        joints,
        model,
        geometry_samples=args.geometry_samples,
        seed=args.seed,
    )

    np.savez_compressed(
        output_path,
        state_probability=predictions["state_probability"],
        state_names=predictions["state_names"],
        side_names=predictions["side_names"],
    )

    print(f"Saved contact probabilities for {len(joints)} frames to {output_path}")


if __name__ == "__main__":
    main()
