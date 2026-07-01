#!/usr/bin/env python3
"""Create a compact 2D BraTS-style subset from TCGA NIfTI volumes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed  # noqa: E402

MODALITIES = ("t1", "t1gd", "t2", "flair")
MODALITY_SUFFIXES = {
    "t1": "_t1.nii.gz",
    "t1gd": "_t1Gd.nii.gz",
    "t2": "_t2.nii.gz",
    "flair": "_flair.nii.gz",
}
SEGMENTATION_SUFFIXES = (
    "_GlistrBoost_ManuallyCorrected.nii.gz",
    "_GlistrBoost.nii.gz",
)


def _find_file(case_dir: Path, suffixes: tuple[str, ...] | list[str]) -> Path | None:
    for suffix in suffixes:
        matches = sorted(case_dir.glob(f"*{suffix}"))
        if matches:
            return matches[0]
    return None


def _case_files(case_dir: Path) -> dict[str, Path] | None:
    files: dict[str, Path] = {}
    for modality, suffix in MODALITY_SUFFIXES.items():
        path = _find_file(case_dir, [suffix])
        if path is None:
            return None
        files[modality] = path

    mask_path = _find_file(case_dir, list(SEGMENTATION_SUFFIXES))
    if mask_path is None:
        return None
    files["mask"] = mask_path
    return files


def _load_volume(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)


def _robust_normalize(volume: np.ndarray) -> np.ndarray:
    foreground = volume[volume > 0]
    if foreground.size < 16:
        return np.zeros_like(volume, dtype=np.float32)

    lo, hi = np.percentile(foreground, (1, 99))
    if hi <= lo:
        hi = float(foreground.max())
        lo = float(foreground.min())
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)

    normalized = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
    normalized[volume <= 0] = 0.0
    return normalized.astype(np.float32)


def _take_slice(volume: np.ndarray, index: int, axis: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    channels = [
        cv2.resize(image[..., channel], (size, size), interpolation=cv2.INTER_LINEAR)
        for channel in range(image.shape[-1])
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


def _select_slices(
    mask: np.ndarray,
    flair: np.ndarray,
    axis: int,
    min_tumor_pixels: int,
    max_positive: int,
    max_negative: int,
) -> list[tuple[int, bool]]:
    tumor_counts = []
    brain_counts = []
    for index in range(mask.shape[axis]):
        mask_slice = _take_slice(mask, index, axis)
        flair_slice = _take_slice(flair, index, axis)
        tumor_counts.append((index, int(mask_slice.sum())))
        brain_counts.append((index, int((flair_slice > 0).sum())))

    positives = [(idx, count) for idx, count in tumor_counts if count >= min_tumor_pixels]
    positives = sorted(positives, key=lambda item: item[1], reverse=True)[:max_positive]
    positive_indices = {idx for idx, _ in positives}

    negatives: list[int] = []
    if positives and max_negative > 0:
        lo = max(0, min(positive_indices) - 8)
        hi = min(mask.shape[axis] - 1, max(positive_indices) + 8)
        tumor_by_index = dict(tumor_counts)
        candidates = [
            idx
            for idx, brain_count in brain_counts
            if (
                lo <= idx <= hi
                and idx not in positive_indices
                and tumor_by_index[idx] == 0
                and brain_count >= min_tumor_pixels * 4
            )
        ]
        if len(candidates) < max_negative:
            global_candidates = [
                idx
                for idx, brain_count in brain_counts
                if (
                    idx not in positive_indices
                    and idx not in candidates
                    and tumor_by_index[idx] == 0
                    and brain_count >= min_tumor_pixels * 4
                )
            ]
            candidates.extend(global_candidates)

        if candidates:
            positions = np.linspace(0, len(candidates) - 1, min(max_negative, len(candidates))).round().astype(int)
            negatives = [candidates[pos] for pos in positions]

    selected = [(idx, True) for idx in sorted(positive_indices)]
    selected.extend((idx, False) for idx in negatives)
    return selected


def _prepare_output(output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    if output_dir.exists() and overwrite:
        for child in ("images", "masks", "previews", "manifest.json", "summary.json"):
            target = output_dir / child
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    previews_dir = output_dir / "previews"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, masks_dir, previews_dir


def prepare_brats_subset(
    raw_dir: Path,
    output_dir: Path,
    image_size: int,
    max_cases: int,
    max_slices_per_case: int,
    negative_slices_per_case: int,
    min_tumor_pixels: int,
    slice_axis: int,
    overwrite: bool,
    seed: int,
) -> dict[str, Any]:
    images_dir, masks_dir, previews_dir = _prepare_output(output_dir, overwrite)
    rng = np.random.default_rng(seed)

    case_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    rng.shuffle(case_dirs)

    manifest: list[dict[str, Any]] = []
    used_cases = 0
    skipped_cases: list[str] = []

    for case_dir in tqdm(case_dirs, desc="Preparing BraTS cases"):
        if used_cases >= max_cases:
            break

        files = _case_files(case_dir)
        if files is None:
            skipped_cases.append(case_dir.name)
            continue

        try:
            volumes = {modality: _robust_normalize(_load_volume(files[modality])) for modality in MODALITIES}
            raw_mask = _load_volume(files["mask"])
        except Exception as exc:
            skipped_cases.append(f"{case_dir.name}: {exc}")
            continue

        shapes = {volume.shape for volume in volumes.values()}
        shapes.add(raw_mask.shape)
        if len(shapes) != 1:
            skipped_cases.append(f"{case_dir.name}: mismatched shapes {sorted(shapes)}")
            continue

        mask = raw_mask > 0
        if not mask.any():
            skipped_cases.append(f"{case_dir.name}: empty mask")
            continue

        selected = _select_slices(
            mask=mask,
            flair=volumes["flair"],
            axis=slice_axis,
            min_tumor_pixels=min_tumor_pixels,
            max_positive=max_slices_per_case,
            max_negative=negative_slices_per_case,
        )
        if not selected:
            skipped_cases.append(f"{case_dir.name}: no selected slices")
            continue

        used_cases += 1
        for slice_index, has_tumor in selected:
            channels = [_take_slice(volumes[modality], slice_index, slice_axis) for modality in MODALITIES]
            image = np.stack(channels, axis=-1)
            mask_slice = _take_slice(mask, slice_index, slice_axis)

            image = _resize_image(image, image_size)
            mask_out = _resize_mask(mask_slice, image_size)
            sample_id = f"{case_dir.name}_z{slice_index:03d}"

            image_path = images_dir / f"{sample_id}.npy"
            mask_path = masks_dir / f"{sample_id}.png"
            preview_path = previews_dir / f"{sample_id}_flair.png"

            np.save(image_path, image)
            cv2.imwrite(str(mask_path), (mask_out * 255).astype(np.uint8))
            cv2.imwrite(str(preview_path), (image[..., 3] * 255).astype(np.uint8))

            manifest.append(
                {
                    "id": sample_id,
                    "image": str(image_path.relative_to(output_dir)),
                    "mask": str(mask_path.relative_to(output_dir)),
                    "source_case": case_dir.name,
                    "slice_index": int(slice_index),
                    "has_tumor": bool(has_tumor),
                    "tumor_pixels": int(mask_out.sum()),
                    "modalities": list(MODALITIES),
                }
            )

    manifest = sorted(manifest, key=lambda row: row["id"])
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "used_cases": used_cases,
        "samples": len(manifest),
        "modalities": list(MODALITIES),
        "image_size": image_size,
        "max_cases": max_cases,
        "max_slices_per_case": max_slices_per_case,
        "negative_slices_per_case": negative_slices_per_case,
        "min_tumor_pixels": min_tumor_pixels,
        "skipped_cases": skipped_cases[:25],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare compact TCGA/BraTS NIfTI subset")
    parser.add_argument("--config", default="configs/brats.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--max-slices-per-case", type=int, default=16)
    parser.add_argument("--negative-slices-per-case", type=int, default=4)
    parser.add_argument("--min-tumor-pixels", type=int, default=25)
    parser.add_argument("--slice-axis", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    set_seed(config["project"]["seed"])

    data_cfg = config["data"]
    raw_dir = ROOT / (args.raw_dir or data_cfg["raw_dir"])
    output_dir = ROOT / (args.output or data_cfg["data_dir"])

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw NIfTI directory not found: {raw_dir}")

    summary = prepare_brats_subset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        image_size=data_cfg["image_size"],
        max_cases=args.max_cases,
        max_slices_per_case=args.max_slices_per_case,
        negative_slices_per_case=args.negative_slices_per_case,
        min_tumor_pixels=args.min_tumor_pixels,
        slice_axis=args.slice_axis,
        overwrite=args.overwrite,
        seed=config["project"]["seed"],
    )
    print(
        f"Prepared {summary['samples']} slices from {summary['used_cases']} cases "
        f"at {summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
