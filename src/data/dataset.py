"""Production-grade streaming Dataset for medical image segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Loads grayscale PNG image/mask pairs from disk without holding the full dataset in memory."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        train_ratio: float = 0.8,
        transform: Callable | None = None,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = self._load_manifest()

        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.samples))
        rng.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)

        if split == "train":
            selected = indices[:split_idx]
        elif split == "val":
            selected = indices[split_idx:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.samples = [self.samples[i] for i in selected]

    def _load_manifest(self) -> list[dict[str, str]]:
        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)

        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(
                f"No manifest or image/mask folders found in {self.data_dir}. "
                "Run `python scripts/generate_data.py` first."
            )

        samples = []
        for image_path in sorted(images_dir.glob("*.png")):
            mask_path = masks_dir / image_path.name
            if mask_path.exists():
                samples.append(
                    {
                        "id": image_path.stem,
                        "image": str(image_path),
                        "mask": str(mask_path),
                    }
                )
        return samples

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.data_dir / path

    def _load_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        return (image.astype(np.float32) / 255.0).clip(0.0, 1.0)

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")
        return (mask > 0).astype(np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        image = self._load_image(self._resolve_path(sample["image"]))
        mask = self._load_mask(self._resolve_path(sample["mask"]))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            if isinstance(mask, torch.Tensor):
                mask = (mask > 0).long()
            else:
                mask = torch.from_numpy((mask > 0).astype(np.int64))
        else:
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "id": sample["id"],
        }
