#!/usr/bin/env python3
"""Train segmentation model with AMP and OneCycleLR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.factory import build_dataloaders  # noqa: E402
from src.data.synthetic import generate_dataset  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    set_seed(config["project"]["seed"])

    data_dir = ROOT / config["data"]["data_dir"]
    if not (data_dir / "manifest.json").exists():
        print("Dataset not found - generating synthetic data...")
        data_cfg = config["data"]
        generate_dataset(
            output_dir=data_dir,
            num_samples=data_cfg.get("num_samples", 200),
            image_size=data_cfg["image_size"],
            seed=config["project"]["seed"],
            tumor_ratio=data_cfg.get("tumor_ratio", 0.5),
        )

    train_loader, val_loader = build_dataloaders(config)
    trainer = Trainer(config)
    result = trainer.fit(train_loader, val_loader)
    print(f"Training complete. Best validation Dice: {result['best_dice']:.4f}")


if __name__ == "__main__":
    main()
