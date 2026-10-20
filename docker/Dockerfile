# GPU inference image (x86_64 server). For Jetson use an L4T base (nvcr.io/nvidia/l4t-jetpack)
# and rebuild TensorRT engines on the device.
# Build:  docker build -f docker/Dockerfile -t vehicle-detect:latest .
# Run:    docker run --gpus all -p 8000:8000 -v $PWD/models:/app/models vehicle-detect:latest
# CUDA 13.0 matches the cu130 PyTorch wheels (D-013); host driver must be >= 580.
FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) deps layer (cached unless pyproject/uv.lock change)
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra serve --no-install-project

# 2) project code
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra serve

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "vehicle_detect.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
