#!/usr/bin/env python3
"""Export trained PyTorch model to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deployment.export_onnx import export_to_onnx  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    checkpoint = ROOT / args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Run train.py first.")

    output = args.output or config["export"]["onnx_path"]
    path = export_to_onnx(checkpoint, config, ROOT / output)
    print(f"ONNX model exported to {path}")


if __name__ == "__main__":
    main()
