"""Smoke tests for the segmentation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
import json

import cv2
import numpy as np
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


def test_multichannel_dataloader() -> None:
    out = ROOT / "data" / "test_multichannel"
    images = out / "images"
    masks = out / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx in range(4):
        image = np.random.default_rng(idx).random((64, 64, 4), dtype=np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:34, 22:38] = 255
        image_path = images / f"sample_{idx:04d}.npy"
        mask_path = masks / f"sample_{idx:04d}.png"
        np.save(image_path, image)
        cv2.imwrite(str(mask_path), mask)
        manifest.append(
            {
                "id": f"sample_{idx:04d}",
                "image": str(image_path.relative_to(out)),
                "mask": str(mask_path.relative_to(out)),
            }
        )

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    config = load_config(ROOT / "configs/default.yaml")
    config["data"]["data_dir"] = "data/test_multichannel"
    config["data"]["image_size"] = 64
    config["data"]["batch_size"] = 2
    config["model"]["in_channels"] = 4

    train_loader, _ = build_dataloaders(config)
    batch = next(iter(train_loader))
    assert batch["image"].shape[1] == 4
    assert batch["mask"].dtype == torch.int64


if __name__ == "__main__":
    test_synthetic_data_generation()
    test_model_forward()
    test_dataloader()
    test_multichannel_dataloader()
    print("All smoke tests passed.")
