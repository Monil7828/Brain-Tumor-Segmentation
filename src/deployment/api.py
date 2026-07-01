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
    version="1.1.0",
)

_session: ort.InferenceSession | None = None
_image_size: int = 256
_input_channels: int = 1
_tumor_threshold: float = 0.005


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deployment_settings() -> tuple[Path, int, float]:
    root = _project_root()
    model_path = Path(os.environ.get("MODEL_PATH", "checkpoints/model.onnx"))
    if not model_path.is_absolute():
        model_path = root / model_path
    image_size = int(os.environ.get("IMAGE_SIZE", "256"))
    tumor_threshold = float(os.environ.get("TUMOR_THRESHOLD", "0.005"))
    return model_path, image_size, tumor_threshold


def load_model(model_path: str, image_size: int = 256, tumor_threshold: float = 0.005) -> None:
    global _session, _image_size, _input_channels, _tumor_threshold
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _session = ort.InferenceSession(str(path), providers=providers)
    _image_size = image_size
    _tumor_threshold = tumor_threshold

    input_shape = _session.get_inputs()[0].shape
    channels = input_shape[1] if len(input_shape) >= 4 else 1
    _input_channels = int(channels) if isinstance(channels, int) else 1


def _preprocess_grayscale(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(image, dtype=np.float32) / 255.0
    return cv2.resize(arr, (_image_size, _image_size), interpolation=cv2.INTER_LINEAR)


def _normalize_batch(arr: np.ndarray) -> np.ndarray:
    return ((arr - 0.5) / 0.5).astype(np.float32)


def preprocess(image_bytes: bytes) -> tuple[np.ndarray, str | None]:
    arr = _preprocess_grayscale(image_bytes)
    warning = None
    if _input_channels > 1:
        arr = np.repeat(arr[np.newaxis, ...], _input_channels, axis=0)
        warning = "single uploaded image repeated across model channels; use /predict/multimodal for real BraTS input"
    else:
        arr = arr[np.newaxis, ...]
    return _normalize_batch(arr[np.newaxis, ...]), warning


def preprocess_modalities(t1: bytes, t1gd: bytes, t2: bytes, flair: bytes) -> np.ndarray:
    if _input_channels != 4:
        raise HTTPException(status_code=400, detail=f"Loaded model expects {_input_channels} input channel(s), not 4")
    channels = [_preprocess_grayscale(payload) for payload in (t1, t1gd, t2, flair)]
    arr = np.stack(channels, axis=0)
    return _normalize_batch(arr[np.newaxis, ...])


def postprocess(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = _softmax(logits[0])
    mask = np.argmax(probs, axis=0).astype(np.uint8)
    confidence = probs.max(axis=0)
    return mask, confidence


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)


def _run_inference(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if _session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.perf_counter()
    outputs = _session.run(None, {"input": tensor.astype(np.float32)})
    latency_ms = (time.perf_counter() - start) * 1000
    mask, confidence = postprocess(outputs[0])
    return mask, confidence, latency_ms


def _prediction_payload(
    mask: np.ndarray,
    confidence: np.ndarray,
    latency_ms: float,
    warning: str | None = None,
) -> dict[str, Any]:
    tumor_pixels = int((mask == 1).sum())
    total_pixels = int(mask.size)
    tumor_ratio = tumor_pixels / total_pixels
    payload: dict[str, Any] = {
        "latency_ms": round(latency_ms, 2),
        "tumor_pixel_ratio": round(tumor_ratio, 4),
        "tumor_detected": tumor_ratio >= _tumor_threshold,
        "tumor_threshold": _tumor_threshold,
        "mean_confidence": round(float(confidence.mean()), 4),
        "image_size": _image_size,
        "input_channels": _input_channels,
    }
    if warning is not None:
        payload["warning"] = warning
    return payload


@app.on_event("startup")
async def startup() -> None:
    if _session is not None:
        return
    model_path, image_size, tumor_threshold = _deployment_settings()
    if model_path.exists():
        load_model(str(model_path), image_size=image_size, tumor_threshold=tumor_threshold)


@app.get("/health")
async def health() -> dict[str, str]:
    status = "ready" if _session is not None else "model_not_loaded"
    return {"status": status, "input_channels": str(_input_channels)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    contents = await file.read()
    tensor, warning = preprocess(contents)
    mask, confidence, latency_ms = _run_inference(tensor)
    return JSONResponse(_prediction_payload(mask, confidence, latency_ms, warning))


@app.post("/predict/mask")
async def predict_mask(file: UploadFile = File(...)) -> Response:
    contents = await file.read()
    tensor, _ = preprocess(contents)
    mask, _, _ = _run_inference(tensor)

    mask_image = (mask * 255).astype(np.uint8)
    _, encoded = cv2.imencode(".png", mask_image)
    return Response(content=encoded.tobytes(), media_type="image/png")


@app.post("/predict/multimodal")
async def predict_multimodal(
    t1: UploadFile = File(...),
    t1gd: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
) -> JSONResponse:
    tensor = preprocess_modalities(
        await t1.read(),
        await t1gd.read(),
        await t2.read(),
        await flair.read(),
    )
    mask, confidence, latency_ms = _run_inference(tensor)
    return JSONResponse(_prediction_payload(mask, confidence, latency_ms))


@app.post("/predict/multimodal/mask")
async def predict_multimodal_mask(
    t1: UploadFile = File(...),
    t1gd: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
) -> Response:
    tensor = preprocess_modalities(
        await t1.read(),
        await t1gd.read(),
        await t2.read(),
        await flair.read(),
    )
    mask, _, _ = _run_inference(tensor)

    mask_image = (mask * 255).astype(np.uint8)
    _, encoded = cv2.imencode(".png", mask_image)
    return Response(content=encoded.tobytes(), media_type="image/png")
