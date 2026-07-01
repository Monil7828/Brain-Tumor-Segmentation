"""Synthetic BraTS-style MRI data generator for demo and CI runs."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _draw_brain_outline(size: int) -> np.ndarray:
    """Create an elliptical brain-shaped background."""
    image = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, size // 2)
    axes = (int(size * 0.42), int(size * 0.48))
    cv2.ellipse(image, center, axes, 0, 0, 360, 0.35, -1)
    return image


def _add_tumor(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Add a random blob tumor and return image + binary mask."""
    size = image.shape[0]
    mask = np.zeros((size, size), dtype=np.float32)

    num_tumors = int(rng.integers(1, 3))
    for _ in range(num_tumors):
        cx = int(rng.integers(size * 0.3, size * 0.7))
        cy = int(rng.integers(size * 0.25, size * 0.75))
        radius = int(rng.integers(size * 0.04, size * 0.12))
        intensity = float(rng.uniform(0.55, 0.95))
        cv2.circle(image, (cx, cy), radius, intensity, -1)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)

        if rng.random() > 0.5:
            offset_x = int(rng.integers(-radius // 2, radius // 2))
            offset_y = int(rng.integers(-radius // 2, radius // 2))
            sub_r = max(2, radius // 2)
            cv2.circle(image, (cx + offset_x, cy + offset_y), sub_r, intensity * 0.85, -1)
            cv2.circle(mask, (cx + offset_x, cy + offset_y), sub_r, 1.0, -1)

    return image, mask


def _add_noise_and_texture(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, 0.03, image.shape).astype(np.float32)
    image = np.clip(image + noise, 0.0, 1.0)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return np.clip(0.7 * image + 0.3 * blur, 0.0, 1.0)


def generate_sample(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = _draw_brain_outline(size)
    image, mask = _add_tumor(image, rng)
    image = _add_noise_and_texture(image, rng)
    return image, mask


def generate_dataset(
    output_dir: str | Path,
    num_samples: int = 200,
    image_size: int = 256,
    seed: int = 42,
) -> Path:
    """Write synthetic grayscale MRI-style images and masks to disk."""
    output = Path(output_dir)
    images_dir = output / "images"
    masks_dir = output / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    manifest = []

    for idx in range(num_samples):
        image, mask = generate_sample(image_size, rng)
        stem = f"sample_{idx:04d}"
        image_path = images_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.png"

        cv2.imwrite(str(image_path), (image * 255).astype(np.uint8))
        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        manifest.append({"id": stem, "image": str(image_path), "mask": str(mask_path)})

    with open(output / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return output
