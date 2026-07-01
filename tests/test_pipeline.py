"""Smoke tests for the segmentation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import generate_dataset  # noqa: E402
from src.data.factory import build_dataloaders  # noqa: E402
from src.models.unet import build_model  # noqa: E402
from src.utils import load_config  # noqa: E402


def test_synthetic_data_generation(tmp_dir: Path | None = None) -> None:
    out = ROOT / "data" / "test_processed"
    generate_dataset(out, num_samples=10, image_size=64, seed=0)
    assert (out / "manifest.json").exists()
    assert len(list((out / "images").glob("*.png"))) == 10


def test_model_forward() -> None:
    model = build_model(in_channels=1, num_classes=2, base_channels=16)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 2, 64, 64)


def test_dataloader() -> None:
    config = load_config(ROOT / "configs/default.yaml")
    config["data"]["data_dir"] = "data/test_processed"
    config["data"]["image_size"] = 64
    config["data"]["batch_size"] = 2

    out = ROOT / config["data"]["data_dir"]
    if not (out / "manifest.json").exists():
        generate_dataset(out, num_samples=10, image_size=64, seed=0)

    train_loader, val_loader = build_dataloaders(config)
    batch = next(iter(train_loader))
    assert batch["image"].shape[0] <= 2
    assert batch["mask"].dtype == torch.int64


if __name__ == "__main__":
    test_synthetic_data_generation()
    test_model_forward()
    test_dataloader()
    print("All smoke tests passed.")
