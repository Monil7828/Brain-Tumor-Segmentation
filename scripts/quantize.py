#!/usr/bin/env python3
"""Apply Post-Training Static Quantization (PTQ)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.factory import build_dataloaders  # noqa: E402
from src.optimization.quantization import QuantizationEngine  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize trained model to INT8")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    set_seed(config["project"]["seed"])

    checkpoint = ROOT / args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Run train.py first.")

    _, val_loader = build_dataloaders(config)
    engine = QuantizationEngine(config)

    output_path = ROOT / config["quantization"]["output_path"]
    engine.quantize(checkpoint, val_loader, output_path)

    size_info = engine.compare_size(checkpoint, output_path)
    print(f"Quantized model saved to {output_path}")
    print(
        f"Size: FP32={size_info['fp32_mb']} MB -> INT8={size_info['int8_mb']} MB "
        f"({size_info['reduction_ratio']}x reduction)"
    )


if __name__ == "__main__":
    main()
