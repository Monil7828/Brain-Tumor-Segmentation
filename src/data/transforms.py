"""Albumentations-based augmentation pipelines for segmentation."""

from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def _channel_stats(in_channels: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (0.5,) * in_channels, (0.5,) * in_channels


def build_train_transforms(image_size: int, aug_cfg: dict[str, Any], in_channels: int = 1) -> A.Compose:
    mean, std = _channel_stats(in_channels)
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
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )


def build_val_transforms(image_size: int, in_channels: int = 1) -> A.Compose:
    mean, std = _channel_stats(in_channels)
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )
