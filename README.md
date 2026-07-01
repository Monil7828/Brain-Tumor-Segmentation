# End-to-End Brain Tumor Segmentation Pipeline

Production-style PyTorch project for brain tumor **segmentation** on synthetic MRI-style images. It covers data generation, U-Net training, ONNX export, FastAPI serving, Docker deployment, and Render hosting.

**Medical disclaimer:** This is a portfolio/research engineering demo, not a clinically validated diagnostic product.

## How It Works

### Model

| Item | Value |
| --- | --- |
| Architecture | U-Net (`src/models/unet.py`) |
| Input | 1-channel grayscale MRI, resized to 256×256 |
| Output | 2-class pixel logits: background (0) and tumor (1) |
| Deployed format | ONNX via ONNX Runtime (`checkpoints/model.onnx`) |

### Training data

The model is trained on **synthetic** images generated in `src/data/synthetic.py`:

- ~50% samples contain bright circular tumor blobs
- ~50% samples are tumor-free (brain outline + noise only)
- Loss: 50% CrossEntropy + 50% Dice
- Metrics: **Dice** and **IoU** on validation split

Because training uses synthetic shapes—not real BraTS or clinical MRI—the model will not generalize perfectly to arbitrary real-world scans.

### Segmentation

For each uploaded image the API:

1. Converts to grayscale and resizes to 256×256
2. Normalizes pixels to `(x - 0.5) / 0.5`
3. Runs ONNX inference to get per-pixel class probabilities
4. Builds a mask where `P(tumor) >= 0.5`

### Tumor detection (`tumor_detected`)

Detection is **derived from the segmentation mask**, not a separate classifier. A scan is flagged as tumor-positive only when **all three** conditions pass:

| Rule | Default | Meaning |
| --- | --- | --- |
| `tumor_pixel_ratio >= tumor_threshold` | **1.0%** | At least 1% of pixels are confident tumor |
| `largest_tumor_region_ratio >= min_tumor_region_ratio` | **1.5%** | Largest connected tumor blob is ≥1.5% of image |
| `tumor_confidence >= tumor_confidence_threshold` | **50%** | Average tumor-class probability on detected pixels ≥0.5 |

There is no single "tumor percentage" score—the API returns the ratios above plus `tumor_confidence` (mean softmax probability for tumor class on flagged pixels).

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --epochs 10
python scripts/serve.py
```

API docs: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Output |
| --- | --- | --- |
| `GET` | `/health` | `{"status": "ready"}` or `"model_not_loaded"` |
| `POST` | `/predict` | JSON with segmentation-derived metrics (see below) |
| `POST` | `/predict/mask` | PNG binary mask (white = predicted tumor) |

### `/predict` response example

```json
{
  "latency_ms": 65.2,
  "tumor_pixel_ratio": 0.2236,
  "largest_tumor_region_ratio": 0.2236,
  "tumor_confidence": 0.9213,
  "tumor_detected": true,
  "tumor_threshold": 0.01,
  "tumor_confidence_threshold": 0.5,
  "min_tumor_region_ratio": 0.015,
  "mean_confidence": 0.9069,
  "image_size": 256
}
```

## Deploy on Render

1. Push this repo to GitHub.
2. In [Render Dashboard](https://dashboard.render.com), click **New → Blueprint**.
3. Connect the repo—Render reads `render.yaml` at the repo root.
4. Render builds `deploy/Dockerfile`, which copies the committed ONNX model and starts the FastAPI server.
5. Health check: `GET /health`
6. Test inference: `POST /predict` with a multipart image upload.

Environment variables (set automatically by `render.yaml`):

| Variable | Default |
| --- | --- |
| `MODEL_PATH` | `checkpoints/model.onnx` |
| `IMAGE_SIZE` | `256` |
| `TUMOR_THRESHOLD` | `0.01` |
| `TUMOR_CONFIDENCE_THRESHOLD` | `0.5` |
| `MIN_TUMOR_REGION_RATIO` | `0.015` |

### Local Docker

```bash
docker compose -f deploy/docker-compose.yml up --build
```

## Project Structure

```text
configs/default.yaml   # Training + deployment settings
deploy/                # Dockerfile, docker-compose, inference deps
scripts/               # generate_data, train, export, serve, run_pipeline
src/data/              # Synthetic dataset + loaders
src/models/            # U-Net
src/training/          # Trainer, losses, metrics
src/deployment/        # FastAPI + ONNX export
tests/                 # Smoke tests
checkpoints/model.onnx # Deployed model artifact
render.yaml            # Render Blueprint
```

## Tests

```bash
python tests/test_pipeline.py
```
