.DEFAULT_GOAL := help
.PHONY: help setup dvc-init gpu-check lint fmt test download prepare train train-5070 train-resume register export eval bench count serve mlflow repro notebook clean

TRAIN_CONFIG ?= configs/train/rfdetr_small.yaml
DVC_REMOTE ?= $(HOME)/dvc-store

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Install deps (+data tools) and git hooks
	@test -d .git || git init -q
	uv sync --all-groups
	uv run pre-commit install
	uv run nbstripout --install --attributes .gitattributes
	@test -f .env || cp .env.example .env

lint: ## Ruff lint + format check (src, tests, notebooks)
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Auto-fix lint and format
	uv run ruff check --fix .
	uv run ruff format .

test: ## Unit tests (CPU only)
	uv run pytest -m "not gpu"

download: ## HF / Kaggle / Roboflow -> data/raw/<source> (configs/data/sources.yaml)
	uv run vd-download

prepare: ## Raw datasets -> data/processed/coco
	uv run vd-prepare

dvc-init: ## DVC + local/NAS remote (D-015). Override: make dvc-init DVC_REMOTE=/mnt/nas/dvc-store
	@test -d .dvc || uv run dvc init -q
	mkdir -p $(DVC_REMOTE)
	uv run dvc remote add -d storage $(DVC_REMOTE) --local -f
	uv run dvc remote list

gpu-check: ## Verify CUDA/torch can run kernels on this GPU (RTX 5070 = sm_120)
	nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
	uv run vd-gpu-check

train: ## Fine-tune RF-DETR (logs to MLflow). Override: make train TRAIN_CONFIG=...
	uv run vd-train --config $(TRAIN_CONFIG)

train-5070: gpu-check ## Fine-tune RF-DETR Small on RTX 5070 12GB (bf16, eff. batch 16)
	uv run vd-train --config configs/train/rfdetr_small_rtx5070.yaml

train-resume: ## Resume from <output_dir>/last.ckpt. Override: make train-resume TRAIN_CONFIG=...
	uv run vd-train --config $(TRAIN_CONFIG) --resume auto

register: ## Register a run in MLflow Model Registry: make register RUN_ID=<id> [ALIAS=candidate]
	uv run vd-register --run-id $(RUN_ID) --alias $(or $(ALIAS),candidate)

export: ## Export checkpoint -> ONNX / TensorRT
	uv run vd-export

eval: ## mAP, per-class AP, per-condition mAP
	uv run vd-evaluate

bench: ## End-to-end 60 FPS gate (run on the target device)
	uv run vd-bench

count: ## Line-crossing vehicle count demo video
	uv run vd-count

serve: ## FastAPI inference service on :8000
	uv run --extra serve uvicorn vehicle_detect.serve.app:app --host 0.0.0.0 --port 8000

mlflow: ## MLflow UI on :5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

repro: ## Re-run the DVC pipeline (only changed stages)
	uv run dvc repro

notebook: ## Start JupyterLab
	uv run jupyter lab notebooks/

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__
