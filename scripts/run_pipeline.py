#!/usr/bin/env python3
"""Run the full end-to-end pipeline: data -> train -> export."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full segmentation pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    config_flag = ["--config", args.config]
    if not args.skip_data:
        run_step("Step 1: Generate Data", [python, "scripts/generate_data.py", *config_flag])

    train_cmd = [python, "scripts/train.py", *config_flag]
    if args.epochs:
        train_cmd.extend(["--epochs", str(args.epochs)])
    run_step("Step 2: Train Model (AMP + OneCycleLR)", train_cmd)
    run_step("Step 3: Export to ONNX", [python, "scripts/export_onnx.py", *config_flag])

    print("\nPipeline complete!")
    print("  - Checkpoints: checkpoints/")
    print("  - ONNX model:  checkpoints/model.onnx")
    print("  - Serve API:   uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
