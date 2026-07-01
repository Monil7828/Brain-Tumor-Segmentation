"""ONNX export utilities for cross-platform deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.unet import build_model


def export_to_onnx(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    output_path: str | Path,
) -> Path:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = config["model"]
    model = build_model(
        in_channels=model_cfg["in_channels"],
        num_classes=config["data"]["num_classes"],
        base_channels=model_cfg["base_channels"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image_size = config["data"]["image_size"]
    dummy = torch.randn(1, model_cfg["in_channels"], image_size, image_size)

    dynamic_axes = None
    if config["export"].get("dynamic_batch", True):
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": max(config["export"]["opset_version"], 18),
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": dynamic_axes,
        "dynamo": False,
    }

    torch.onnx.export(model, dummy, str(output), **export_kwargs)
    return output
