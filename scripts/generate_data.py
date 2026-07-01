#!/usr/bin/env python3
"""Generate synthetic MRI-style dataset for local development."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import generate_dataset  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic MRI segmentation data")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--output", default=None, help="Override output directory")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    set_seed(config["project"]["seed"])

    output_dir = args.output or config["data"]["data_dir"]
    data_cfg = config["data"]
    path = generate_dataset(
        output_dir=ROOT / output_dir,
        num_samples=args.num_samples or data_cfg.get("num_samples", 200),
        image_size=data_cfg["image_size"],
        seed=config["project"]["seed"],
        tumor_ratio=data_cfg.get("tumor_ratio", 0.5),
    )
    print(f"Generated {args.num_samples} samples at {path}")


if __name__ == "__main__":
    main()
