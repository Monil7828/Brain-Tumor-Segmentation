# Train and export ONNX (skipped when checkpoints/model.onnx is already present).
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN if [ ! -f checkpoints/model.onnx ]; then \
      python scripts/generate_data.py --num-samples 200 && \
      python scripts/train.py --epochs 5 && \
      python scripts/export_onnx.py; \
    fi

# Slim runtime image for serving (no PyTorch).
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY --from=builder /app/checkpoints/ ./checkpoints/

ENV MODEL_PATH=checkpoints/model.onnx \
    IMAGE_SIZE=256 \
    TUMOR_THRESHOLD=0.005

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.deployment.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
