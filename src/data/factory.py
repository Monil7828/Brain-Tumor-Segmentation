"""DataLoader factory with optimal PyTorch settings."""

from __future__ import annotations

import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SegmentationDataset
from src.data.transforms import build_train_transforms, build_val_transforms


def _default_num_workers(configured: int) -> int:
    """Windows multiprocessing in DataLoader requires spawn; 0 is safest for demos."""
    if sys.platform == "win32":
        return 0
    return configured


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    seed = config["project"]["seed"]
    num_workers = _default_num_workers(data_cfg["num_workers"])

    train_ds = SegmentationDataset(
        data_dir=data_cfg["data_dir"],
        split="train",
        train_ratio=data_cfg["train_split"],
        transform=build_train_transforms(data_cfg["image_size"], config["augmentation"]["train"]),
        seed=seed,
    )
    val_ds = SegmentationDataset(
        data_dir=data_cfg["data_dir"],
        split="val",
        train_ratio=data_cfg["train_split"],
        transform=build_val_transforms(data_cfg["image_size"]),
        seed=seed,
    )

    pin_memory = data_cfg["pin_memory"] and torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": data_cfg["batch_size"],
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader
