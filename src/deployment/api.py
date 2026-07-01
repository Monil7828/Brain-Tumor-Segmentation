"""FastAPI inference server for ONNX segmentation model."""

from __future__ import annotations

import io
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
    description="Production ONNX inference server for medical image segmentation",
    version="1.0.0",
)

_session: ort.InferenceSession | None = None
_image_size: int = 256


def load_model(model_path: str, image_size: int = 256) -> None:
    global _session, _image_size
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _session = ort.InferenceSession(str(path), providers=providers)
    _image_size = image_size


def preprocess(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = cv2.resize(arr, (_image_size, _image_size), interpolation=cv2.INTER_LINEAR)
    arr = (arr - 0.5) / 0.5
    return arr[np.newaxis, np.newaxis, ...]


def postprocess(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = _softmax(logits[0])
    mask = np.argmax(probs, axis=0).astype(np.uint8)
    confidence = probs.max(axis=0)
    return mask, confidence


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)


@app.on_event("startup")
async def startup() -> None:
    model_path = Path(__file__).resolve().parents[2] / "checkpoints" / "model.onnx"
    if model_path.exists():
        load_model(str(model_path))


@app.get("/health")
async def health() -> dict[str, str]:
    status = "ready" if _session is not None else "model_not_loaded"
    return {"status": status}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if _session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    tensor = preprocess(contents)

    start = time.perf_counter()
    outputs = _session.run(None, {"input": tensor.astype(np.float32)})
    latency_ms = (time.perf_counter() - start) * 1000

    mask, confidence = postprocess(outputs[0])
    tumor_pixels = int((mask == 1).sum())
    total_pixels = int(mask.size)
    tumor_ratio = round(tumor_pixels / total_pixels, 4)

    return JSONResponse(
        {
            "latency_ms": round(latency_ms, 2),
            "tumor_pixel_ratio": tumor_ratio,
            "mean_confidence": round(float(confidence.mean()), 4),
            "image_size": _image_size,
        }
    )


@app.post("/predict/mask")
async def predict_mask(file: UploadFile = File(...)) -> Response:
    if _session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    tensor = preprocess(contents)
    outputs = _session.run(None, {"input": tensor.astype(np.float32)})
    mask, _ = postprocess(outputs[0])

    mask_image = (mask * 255).astype(np.uint8)
    _, encoded = cv2.imencode(".png", mask_image)
    return Response(content=encoded.tobytes(), media_type="image/png")
