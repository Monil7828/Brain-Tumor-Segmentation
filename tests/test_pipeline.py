"""Smoke tests for the segmentation pipeline."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import generate_dataset  # noqa: E402
from src.data.factory import build_dataloaders  # noqa: E402
from src.deployment.api import _prediction_payload, load_model, postprocess  # noqa: E402
from src.models.unet import build_model  # noqa: E402
from src.utils import load_config  # noqa: E402


def test_synthetic_data_generation() -> None:
    out = ROOT / "data" / "test_processed"
    generate_dataset(out, num_samples=10, image_size=64, seed=0, tumor_ratio=0.5)
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
        generate_dataset(out, num_samples=10, image_size=64, seed=0, tumor_ratio=0.5)

    train_loader, val_loader = build_dataloaders(config)
    batch = next(iter(train_loader))
    assert batch["image"].shape[0] <= 2
    assert batch["mask"].dtype == torch.int64


def test_detection_on_empty_mask() -> None:
    mask = np.zeros((256, 256), dtype=np.uint8)
    tumor_probs = np.full((256, 256), 0.1, dtype=np.float32)
    confidence = np.full((256, 256), 0.9, dtype=np.float32)
    payload = _prediction_payload(mask, tumor_probs, confidence, latency_ms=1.0)
    assert payload["tumor_detected"] is False


def test_onnx_inference_if_model_exists() -> None:
    model_path = ROOT / "checkpoints" / "model.onnx"
    if not model_path.exists():
        return

    load_model(str(model_path))
    blank = Image.new("L", (256, 256), color=0)
    buffer = io.BytesIO()
    blank.save(buffer, format="PNG")
    from src.deployment.api import preprocess, _run_inference

    tensor = preprocess(buffer.getvalue())
    mask, tumor_probs, confidence, _ = _run_inference(tensor)
    payload = _prediction_payload(mask, tumor_probs, confidence, latency_ms=1.0)
    assert "tumor_detected" in payload


if __name__ == "__main__":
    test_synthetic_data_generation()
    test_model_forward()
    test_dataloader()
    test_detection_on_empty_mask()
    test_onnx_inference_if_model_exists()
    print("All smoke tests passed.")
