"""Evaluation metrics for segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = torch.argmax(logits, dim=1)
    scores = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        if union == 0:
            continue
        scores.append(((2 * intersection + 1e-6) / (union + 1e-6)).item())
    return float(sum(scores) / max(len(scores), 1))


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = torch.argmax(logits, dim=1)
    scores = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        if union == 0:
            continue
        scores.append(((intersection + 1e-6) / (union + 1e-6)).item())
    return float(sum(scores) / max(len(scores), 1))
