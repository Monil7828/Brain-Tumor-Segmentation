#!/usr/bin/env python3
"""Start FastAPI inference server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deployment.api import load_model  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Start segmentation API server")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    deploy_cfg = config["deployment"]
    model_path = args.model or deploy_cfg["model_path"]

    load_model(
        str(ROOT / model_path),
        image_size=deploy_cfg["image_size"],
        tumor_threshold=deploy_cfg.get("tumor_threshold", 0.01),
        tumor_confidence_threshold=deploy_cfg.get("tumor_confidence_threshold", 0.5),
        min_tumor_region_ratio=deploy_cfg.get("min_tumor_region_ratio", 0.015),
    )

    uvicorn.run(
        "src.deployment.api:app",
        host=deploy_cfg["host"],
        port=deploy_cfg["port"],
        reload=False,
    )


if __name__ == "__main__":
    main()
