# vehicle-detect

Real-time vehicle detection + classification (9 Thai traffic classes), target **≥ 60 FPS end-to-end**,
built as a full ML lifecycle project: data → train → evaluate → export → benchmark → serve → monitor.

Stack: **uv** (env/deps) · **ruff** (lint/format incl. notebooks) · **RF-DETR** (Apache-2.0) ·
**supervision + trackers** (ByteTrack, line counting) · **ONNX Runtime / TensorRT** · **MLflow** · **DVC** ·
**FiftyOne** · **FastAPI** · **pre-commit** · **GitHub Actions**

## Quick start

```bash
make setup          # uv sync --all-groups + pre-commit + nbstripout + .env
make help           # list all commands
make notebook       # JupyterLab
```

First run creates `uv.lock`, so **commit `uv.lock`** for reproducible installs.
PyTorch: on macOS you get the default PyPI wheels (Apple Silicon → MPS); on Linux, wheels come
from the CUDA index set in `pyproject.toml` (`cu128`). Change it to match `nvidia-smi`.

## Training on NVIDIA RTX 5070 (12 GB, CUDA)

The RTX 5070 is a Blackwell card (sm_120), so it needs the **CUDA 13 (`cu130`)** PyTorch wheels, which `pyproject.toml` already uses on Linux.
Driver **≥ 580**. On a Windows PC, use **WSL2 + Ubuntu 24.04** (install the NVIDIA driver on Windows only, not inside WSL).
Inside WSL, `uv`, `make`, DVC and TensorRT all work the same as on Linux.

```bash
# inside WSL2 / Ubuntu, repo cloned under ~/ (not /mnt/c — much faster I/O)
nvidia-smi                 # driver >= 580, GPU visible
make setup                 # uv sync pulls torch cu130
make gpu-check             # must print: OK: CUDA kernels run on this GPU (sm_120)
make train-5070            # RF-DETR Small, bf16, batch 4 x accum 4 = effective 16
make train TRAIN_CONFIG=configs/train/rfdetr_medium_rtx5070.yaml   # Medium, gradient checkpointing
```

If you hit OOM, set `batch_size: 2` and `grad_accum_steps: 8`, or turn on `gradient_checkpointing`.
The error "no kernel image is available" means torch is a wrong build, so reinstall with `uv sync --reinstall-package torch`.
The 5070 can also run `make bench`, but if the deployment device is a Jetson, the gate has to be measured on the Jetson.

## Working with Claude Code (harness)

- `CLAUDE.md`: rules for the agent (session protocol, Karpathy coding + training guidelines, guardrails)
- `PLAN.md`: phases and sub-tasks (`phase-1-a` …), each with a **Verify** criterion
- `PROGRESS.md`: current state snapshot (overwritten) + append-only session log
- `DECISIONS.md`: decision records (`D-001` …), the *why* behind choices
- `init.sh`: start-of-session check (sync, lint, tests, next sub-task)

Typical prompt: *"Do phase-3-c"* or *"Continue with the next sub-task in PLAN.md."*

## Project layout

```
configs/            YAML for data / train / export / bench / count — no hardcoded params
data/               DVC-tracked (raw → processed/coco/{train,valid,test})
notebooks/          exploration only — import from src/, never write to data/processed
src/vehicle_detect/ production code (CLI: vd-prepare, vd-train, vd-export, vd-evaluate, vd-bench, vd-count)
tests/              pytest (CPU); `-m gpu` tests run on the target device
docker/             GPU inference image
dvc.yaml            pipeline: prepare → train → export → evaluate → bench
```

## ML lifecycle commands

| Stage | Command | Output |
|---|---|---|
| Download | `make download` | `data/raw/<source>/` from HF / Kaggle / Roboflow (`configs/data/sources.yaml`) |
| Data | `make prepare` | `data/processed/coco`, `reports/data_stats.json` |
| Train | `make train` | `runs/…/checkpoint_best_total.pth`, MLflow run |
| Resume | `make train-resume TRAIN_CONFIG=...` | continues from `last.ckpt` |
| Experiments | `make mlflow` | UI at http://localhost:5000 |
| Register | `make register RUN_ID=<id>` | MLflow Model Registry version, alias `candidate` |
| Export | `make export` | `models/export/*.onnx` (or TensorRT engine **built on the device**) |
| Evaluate | `make eval` | `reports/eval.json` (mAP, per-class AP, per-condition mAP) |
| **60 FPS gate** | `make bench` | `reports/bench.json`, exits 1 if FPS < 60 or p99 > 16.7 ms |
| Business demo | `make count` | annotated video + per-class counts |
| Serve | `make serve` | FastAPI on :8000 (`/detect`, `/health`) |
| Everything | `make repro` | DVC re-runs only changed stages |

## Conventions

- Notebooks are numbered by lifecycle step (`01_` … `04_`). A function used 2+ times moves to `src/`.
- Outputs are stripped on commit (nbstripout). Ruff lints notebooks too.
- Data, weights and videos go in **DVC**, never git (pre-commit blocks files > 1 MB).
- Split by **camera/video**, not by frame (`split.group_by`), or valid/test scores will be inflated.
- FPS is measured **end-to-end** (decode + detect + track) on the **target hardware**, over ≥ 5 min of video.

## Licensing (check before commercial use)

- RF-DETR Nano/Small/Medium: Apache-2.0 ✅
- Ultralytics YOLO26 (`--extra yolo`): AGPL-3.0, including fine-tuned weights. Closed commercial use needs an Enterprise license.
- Datasets: MIO-TCD, KITTI, Cityscapes, Waymo, DAWN are **non-commercial**. BDD100K commercial rights are for members only.
  COCO / Open Images annotations are CC BY 4.0. Production model → fine-tune on your own anonymised Thai CCTV footage.

## Privacy (PDPA)

Process video on the device and send out **counts/events only**, not frames or plates.
ANPR / plate recognition is an add-on that needs a DPIA first.
