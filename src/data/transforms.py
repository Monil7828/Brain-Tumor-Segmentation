"""Albumentations-based augmentation pipelines for segmentation."""

from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def build_train_transforms(image_size: int, aug_cfg: dict[str, Any]) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_prob", 0.5)),
            A.VerticalFlip(p=aug_cfg.get("vertical_flip_prob", 0.3)),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-aug_cfg.get("rotate_limit", 15), aug_cfg.get("rotate_limit", 15)),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get("brightness_limit", 0.2),
                contrast_limit=aug_cfg.get("contrast_limit", 0.2),
                p=0.5,
            ),
            A.GaussNoise(std_range=(0.04, 0.12), p=0.3),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )


def build_val_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )
