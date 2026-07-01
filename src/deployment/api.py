"""FastAPI inference server for ONNX segmentation models."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="ONNX inference server for binary brain tumor segmentation",
    version="1.3.0",
)

_session: ort.InferenceSession | None = None
_image_size: int = 256
_tumor_threshold: float = 0.01
_tumor_confidence_threshold: float = 0.5
_min_tumor_region_ratio: float = 0.015


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deployment_settings() -> tuple[Path, int, float, float, float]:
    root = _project_root()
    model_path = Path(os.environ.get("MODEL_PATH", "checkpoints/model.onnx"))
    if not model_path.is_absolute():
        model_path = root / model_path
    image_size = int(os.environ.get("IMAGE_SIZE", "256"))
    tumor_threshold = float(os.environ.get("TUMOR_THRESHOLD", "0.01"))
    tumor_confidence_threshold = float(os.environ.get("TUMOR_CONFIDENCE_THRESHOLD", "0.5"))
    min_tumor_region_ratio = float(os.environ.get("MIN_TUMOR_REGION_RATIO", "0.015"))
    return model_path, image_size, tumor_threshold, tumor_confidence_threshold, min_tumor_region_ratio


def load_model(
    model_path: str,
    image_size: int = 256,
    tumor_threshold: float = 0.01,
    tumor_confidence_threshold: float = 0.5,
    min_tumor_region_ratio: float = 0.015,
) -> None:
    global _session, _image_size, _tumor_threshold, _tumor_confidence_threshold, _min_tumor_region_ratio
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _session = ort.InferenceSession(str(path), providers=providers)
    _image_size = image_size
    _tumor_threshold = tumor_threshold
    _tumor_confidence_threshold = tumor_confidence_threshold
    _min_tumor_region_ratio = min_tumor_region_ratio


def preprocess(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = cv2.resize(arr, (_image_size, _image_size), interpolation=cv2.INTER_LINEAR)
    arr = arr[np.newaxis, np.newaxis, ...]
    return ((arr - 0.5) / 0.5).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)


def _largest_region_ratio(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(areas.max()) / mask.size


def postprocess(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = _softmax(logits[0])
    tumor_probs = probs[1]
    confident_mask = (tumor_probs >= _tumor_confidence_threshold).astype(np.uint8)
    confidence = probs.max(axis=0)
    return confident_mask, tumor_probs, confidence


def _run_inference(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if _session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.perf_counter()
    outputs = _session.run(None, {"input": tensor.astype(np.float32)})
    latency_ms = (time.perf_counter() - start) * 1000
    mask, tumor_probs, confidence = postprocess(outputs[0])
    return mask, tumor_probs, confidence, latency_ms


def _prediction_payload(
    mask: np.ndarray,
    tumor_probs: np.ndarray,
    confidence: np.ndarray,
    latency_ms: float,
) -> dict[str, Any]:
    tumor_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    tumor_ratio = tumor_pixels / total_pixels
    largest_region_ratio = _largest_region_ratio(mask)

    if tumor_pixels:
        tumor_confidence = float(tumor_probs[mask.astype(bool)].mean())
    else:
        tumor_confidence = float(tumor_probs.max())

    tumor_detected = (
        tumor_ratio >= _tumor_threshold
        and largest_region_ratio >= _min_tumor_region_ratio
        and tumor_confidence >= _tumor_confidence_threshold
    )

    return {
        "latency_ms": round(latency_ms, 2),
        "tumor_pixel_ratio": round(tumor_ratio, 4),
        "largest_tumor_region_ratio": round(largest_region_ratio, 4),
        "tumor_confidence": round(tumor_confidence, 4),
        "tumor_detected": tumor_detected,
        "tumor_threshold": _tumor_threshold,
        "tumor_confidence_threshold": _tumor_confidence_threshold,
        "min_tumor_region_ratio": _min_tumor_region_ratio,
        "mean_confidence": round(float(confidence.mean()), 4),
        "image_size": _image_size,
    }


@app.on_event("startup")
async def startup() -> None:
    if _session is not None:
        return
    model_path, image_size, tumor_threshold, tumor_confidence_threshold, min_tumor_region_ratio = (
        _deployment_settings()
    )
    if model_path.exists():
        load_model(
            str(model_path),
            image_size=image_size,
            tumor_threshold=tumor_threshold,
            tumor_confidence_threshold=tumor_confidence_threshold,
            min_tumor_region_ratio=min_tumor_region_ratio,
        )


@app.get("/health")
async def health() -> dict[str, str]:
    status = "ready" if _session is not None else "model_not_loaded"
    return {"status": status}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    tensor = preprocess(await file.read())
    mask, tumor_probs, confidence, latency_ms = _run_inference(tensor)
    return JSONResponse(_prediction_payload(mask, tumor_probs, confidence, latency_ms))


@app.post("/predict/mask")
async def predict_mask(file: UploadFile = File(...)) -> Response:
    tensor = preprocess(await file.read())
    mask, _, _, _ = _run_inference(tensor)

    mask_image = (mask * 255).astype(np.uint8)
    _, encoded = cv2.imencode(".png", mask_image)
    return Response(content=encoded.tobytes(), media_type="image/png")
