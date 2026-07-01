"""Post-Training Static Quantization (PTQ) for edge deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.unet import build_model
from src.utils import get_device


def _select_qconfig() -> Any:
    if torch.backends.quantized.engine in ("fbgemm", "qnnpack"):
        return torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
    torch.backends.quantized.engine = "fbgemm"
    return torch.ao.quantization.get_default_qconfig("fbgemm")


def _fuse_unet(model: nn.Module) -> None:
    """Fuse Conv-BN-ReLU blocks inside DoubleConv Sequential modules."""
    for module in model.modules():
        if isinstance(module, nn.Sequential) and len(module) >= 3:
            if (
                isinstance(module[0], nn.Conv2d)
                and isinstance(module[1], nn.BatchNorm2d)
                and isinstance(module[2], nn.ReLU)
            ):
                torch.ao.quantization.fuse_modules(
                    module,
                    ["0", "1", "2"],
                    inplace=True,
                )
            if len(module) >= 6 and isinstance(module[3], nn.Conv2d):
                if isinstance(module[4], nn.BatchNorm2d) and isinstance(module[5], nn.ReLU):
                    torch.ao.quantization.fuse_modules(
                        module,
                        ["3", "4", "5"],
                        inplace=True,
                    )


class QuantizationEngine:
    """Apply torch.ao PTQ to reduce model size and inference latency."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = get_device()

    def _build_float_model(self, checkpoint_path: str | Path) -> torch.nn.Module:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_cfg = self.config["model"]
        model = build_model(
            in_channels=model_cfg["in_channels"],
            num_classes=self.config["data"]["num_classes"],
            base_channels=model_cfg["base_channels"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def _quantize_static(
        self,
        float_model: nn.Module,
        calibration_loader: DataLoader,
    ) -> nn.Module:
        _fuse_unet(float_model)
        float_model.qconfig = _select_qconfig()
        prepared = torch.ao.quantization.prepare(float_model, inplace=False)

        num_samples = self.config["quantization"]["calibration_samples"]
        seen = 0
        with torch.no_grad():
            for batch in calibration_loader:
                images = batch["image"]
                prepared(images)
                seen += images.shape[0]
                if seen >= num_samples:
                    break

        return torch.ao.quantization.convert(prepared, inplace=False)

    def _quantize_dynamic(self, float_model: nn.Module) -> nn.Module:
        """Fallback: dynamic quantization on Linear/Conv layers."""
        return torch.ao.quantization.quantize_dynamic(
            float_model,
            {nn.Conv2d, nn.Linear},
            dtype=torch.qint8,
        )

    def quantize(
        self,
        checkpoint_path: str | Path,
        calibration_loader: DataLoader,
        output_path: str | Path,
    ) -> Path:
        float_model = self._build_float_model(checkpoint_path)
        backend = torch.backends.quantized.engine

        try:
            quantized_model = self._quantize_static(float_model, calibration_loader)
            method = "static_ptq"
        except Exception as exc:
            print(f"Static PTQ failed ({exc}); falling back to dynamic quantization.")
            float_model = self._build_float_model(checkpoint_path)
            quantized_model = self._quantize_dynamic(float_model)
            method = "dynamic"

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "quantized_state_dict": quantized_model.state_dict(),
                "config": self.config,
                "backend": backend,
                "method": method,
            },
            output,
        )
        return output

    def compare_size(self, fp32_path: str | Path, int8_path: str | Path) -> dict[str, float]:
        fp32_mb = Path(fp32_path).stat().st_size / (1024 * 1024)
        int8_mb = Path(int8_path).stat().st_size / (1024 * 1024)
        return {
            "fp32_mb": round(fp32_mb, 3),
            "int8_mb": round(int8_mb, 3),
            "reduction_ratio": round(fp32_mb / max(int8_mb, 1e-6), 2),
        }
