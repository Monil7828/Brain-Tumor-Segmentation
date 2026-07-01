"""AMP-enabled training loop with OneCycleLR scheduler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unet import build_model
from src.training.losses import CombinedLoss
from src.training.metrics import dice_score, iou_score
from src.utils import ensure_dir, get_device


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = get_device()
        self.use_amp = config["training"]["amp"] and self.device.type == "cuda"

        model_cfg = config["model"]
        self.model = build_model(
            in_channels=model_cfg["in_channels"],
            num_classes=config["data"]["num_classes"],
            base_channels=model_cfg["base_channels"],
        ).to(self.device)

        train_cfg = config["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.criterion = CombinedLoss()
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.scheduler: torch.optim.lr_scheduler.OneCycleLR | None = None
        self.checkpoint_dir = ensure_dir(train_cfg["checkpoint_dir"])
        self.history: list[dict[str, float]] = []

    def _setup_scheduler(self, train_loader: DataLoader) -> None:
        oc = self.config["training"]["onecycle"]
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config["training"]["learning_rate"],
            epochs=self.config["training"]["epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=oc["pct_start"],
            anneal_strategy=oc["anneal_strategy"],
            div_factor=oc["div_factor"],
            final_div_factor=oc["final_div_factor"],
        )

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
        for step, batch in enumerate(pbar, start=1):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["grad_clip"],
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            batch_dice = dice_score(logits.detach(), masks, self.config["data"]["num_classes"])
            batch_iou = iou_score(logits.detach(), masks, self.config["data"]["num_classes"])
            total_loss += loss.item()
            total_dice += batch_dice
            total_iou += batch_iou

            if step % self.config["training"]["log_interval"] == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice:.4f}")

        n = len(loader)
        return {"loss": total_loss / n, "dice": total_dice / n, "iou": total_iou / n}

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            total_loss += loss.item()
            total_dice += dice_score(logits, masks, self.config["data"]["num_classes"])
            total_iou += iou_score(logits, masks, self.config["data"]["num_classes"])

        n = len(loader)
        return {"loss": total_loss / n, "dice": total_dice / n, "iou": total_iou / n}

    def save_checkpoint(self, epoch: int, metrics: dict[str, float], is_best: bool) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(payload, self.checkpoint_dir / "last.pt")
        if is_best:
            torch.save(payload, self.checkpoint_dir / "best.pt")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        self._setup_scheduler(train_loader)
        best_dice = -1.0
        epochs = self.config["training"]["epochs"]

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
            }
            self.history.append(record)

            is_best = val_metrics["dice"] > best_dice
            if is_best:
                best_dice = val_metrics["dice"]
            self.save_checkpoint(epoch, val_metrics, is_best)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
                f"val_iou={val_metrics['iou']:.4f}"
            )

        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {"best_dice": best_dice, "history_path": str(history_path)}
