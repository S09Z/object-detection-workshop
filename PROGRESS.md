# PROGRESS.md

Two parts:
1. **Current state**: a snapshot that is *overwritten* at the end of every sub-task. A new session reads this first.
2. **Session log**: *append-only*, newest entry at the bottom. Never edit old entries.

Decisions (the *why*) go in `DECISIONS.md`. This file records only *what* and *where we are*.

---

## 1. Current state

> Last updated: 2026-10-20 · by: Claude (phase-1-d)

| Field | Value |
|---|---|
| Current phase | phase-1 — Setup & harness |
| Current sub-task | **phase-1-e**: MLflow tracking |
| Status | `not started` <!-- not started / in progress / blocked / done --> |
| Last done | phase-1-d: DVC initialised, remote `storage` = `~/dvc-store` (local, per machine) |
| Next up | phase-1-e → phase-1-f (needs the RTX 5070 PC) |
| Branch / last commit | development (workflow: commit + push per sub-task; feature branch from origin/development → draft PR to main) |
| Blockers | none |

### Health check (latest `./init.sh`)

| Check | Result | When |
|---|---|---|
| ruff check / format | ✅ pass (project ruff via uv; an older ruff had missed 2 E501 + 1 format diff in convert.py) | 2026-10-20 |
| pytest `-m "not gpu"` | ✅ 20 passed (run without torch) | 2026-10-20 |
| `uv sync --all-groups` | ✅ pass (Mac, Python 3.13 venv) | 2026-10-20 |
| `make gpu-check` (RTX 5070) | ⏳ not run yet | — |

### Key metrics (best so far)

| Metric | Target | Best | MLflow run / DVC rev |
|---|---|---|---|
| test mAP50-95 (9 classes) | > baseline phase-4-a | — | — |
| Count accuracy vs manual count | ≥ 95% (confirm in phase-2-a) | — | — |
| End-to-end FPS (target device) | ≥ 60 | — | — |
| p99 frame time | ≤ 16.7 ms | — | — |

### Model / data in use

| Item | Value |
|---|---|
| Model candidate | — (planned: RF-DETR Small, trained on RTX 5070 with bf16) |
| Production model | — |
| Dataset version (DVC) | — (remote: local disk / NAS `~/dvc-store`, D-015) |
| Target device | RTX 5070 12 GB PC, WSL2 Ubuntu 24.04, CUDA 13 / driver ≥ 580 (D-013, D-014) |
| Real Thai footage | none yet; use public data + freely licensed clips (D-016) |

### Open questions for the user

- [x] Target device → RTX 5070 PC (D-014)
- [x] DVC remote → local disk / NAS (D-015)
- [x] Thai footage → none yet, use public clips (D-016)
- [ ] NAS path, if one exists (otherwise `~/dvc-store`), and how it is backed up (phase-1-d)
- [ ] Number of cameras and resolution for real use (phase-2-b)
- [ ] Confirm the 9 classes and their order (D-003, phase-2-d)
- [x] GitHub repo for phase-1-c → `S09Z/object-detection-workshop`

---

## 2. Session log

Template (copy it for each entry):

```
### YYYY-MM-DD — phase-X-y <short title>
- Status: done | partial | blocked | abandoned
- Done: what changed (files, commands)
- Verify: command run + key result (e.g. mAP50-95 0.61, FPS 74, p99 14.2 ms)
- MLflow run / DVC rev: <id or n/a>
- Decisions: D-00N (see DECISIONS.md) or none
- Notes for next session: anything the next agent must know
```

### 2026-10-20 — project template created
- Status: done
- Done: repo skeleton (uv, ruff, notebooks, DVC pipeline, MLflow, 60 FPS bench gate, CI, Docker), CLAUDE.md, PLAN.md, init.sh, DECISIONS.md
- Verify: `ruff check` + `ruff format --check` clean; `pytest -m "not gpu"` → 9 passed (CPU, without torch)
- MLflow run / DVC rev: n/a
- Decisions: D-001 … D-012
- Notes for next session: `uv.lock` has not been generated (the PyTorch CUDA index was unreachable from the build machine), so run `make setup` on the Mac first

### 2026-10-20 — RTX 5070 training option + user answers
- Status: done
- Done: torch index cu128 → cu130 (PyTorch 2.14 no longer ships cu128); added `configs/train/rfdetr_{small,medium}_rtx5070.yaml`,
  `vd-gpu-check`, `make gpu-check / train-5070 / dvc-init`, Docker base CUDA 13.0.3, phase-1-f in PLAN.md; recorded the target device, DVC remote and footage plan
- Verify: ruff + pytest (CPU) — see the health check above
- MLflow run / DVC rev: n/a
- Decisions: D-013, D-014, D-015, D-016
- Notes for next session: training configs haven't been run on the real 5070 yet; if VRAM overflows, use the fallback in the yaml comments

### 2026-10-20 — dataset download + checkpoint practice
- Status: done
- Done: `vd-download` + `configs/data/sources.yaml` + converters (yolo / coco / hf_metadata → COCO, camera_id from file name);
  `train.py`: `--resume auto`, seed, best_model_metric, run_test, load-test best ckpt, prune archives, log `model/` + git/DVC tags to MLflow;
  `vd-register` (Model Registry + alias); MLflow backend → `sqlite:///mlflow.db`; make `download / train-resume / register`; phase-4-f
- Verify: ruff clean; pytest 20 passed (new: test_checkpoints, test_convert); `uv lock --no-sources` resolved 384 packages
- MLflow run / DVC rev: n/a
- Decisions: D-017, D-018
- Notes for next session: `vd-download` for visaitech hasn't been run against the real repo yet; check the file names against `camera_id_regex`.
  Lineage logging in `train.py` (finding the run by run name) still needs to be tested with a real training run in phase-4-f

### 2026-10-20 — phase-1-a environment + uv.lock
- Status: done
- Done: `uv sync --all-groups` on the Mac created `.venv` and `uv.lock`; `.gitignore` added and `git init` done so the baseline template could be committed first
- Verify: `uv run python -c "import torch, rfdetr, supervision; ..."` printed `2.14.1 True` (MPS)
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: the venv resolved to Python 3.13; phase-1-b pins `.python-version` so the Dockerfile and CI match. The Linux cu130 lock entries are resolved but untested until phase-1-f

### 2026-10-20 — phase-1-b git + pre-commit
- Status: done
- Done: `.python-version` (3.13, matches the lock), `.env.example`, `.gitattributes` (nbstripout), `.pre-commit-config.yaml` (local hooks via uv: ruff check/format, nbstripout, block data/weights files); notebooks stripped; two docstring lines and one format fix in `data/convert.py`
- Verify: `uv run pre-commit run --all-files` all passed; `./init.sh` ended with `OK`
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: the "lint clean" claim in earlier entries came from a different ruff version; the uv-pinned one is authoritative. Still open from the project review: split guards, download delete-before-convert, TensorRT provider options, bench minimum duration

### 2026-10-20 — phase-1-c CI workflow (partial)
- Status: partial (box not ticked)
- Done: `.github/workflows/ci.yml` (job `lint-test`: uv sync --locked, ruff, pytest -m "not gpu")
- Verify: not run — needs a pushed GitHub remote (`gh run list --limit 1` → completed success)
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: after the user adds `origin` and pushes, check the run; `uv sync --locked` on Linux pulls the cu130 torch wheels (large), so consider a CPU-only CI sync if it is slow

### 2026-10-20 — project-review fixes (split guards, safe download)
- Status: done (outside PLAN.md sub-tasks; requested after the review)
- Done: `group_split` scopes groups by `source`, raises if a split would be empty, warns on images without the group key; `vd-download` converts into `.<name>.tmp` and swaps, so a failed convert no longer deletes the existing dataset; tests added (`test_split.py`, `test_download.py`)
- Verify: `uv run pytest -m "not gpu"` → 25 passed; `ruff check` clean
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: still open from the review: TensorRT provider options (fp16, engine cache) vs `format: tensorrt` export, minimum duration check in `vd-bench`, `async def` + blocking inference and no upload limit in serve, Dockerfile pulls torch just to serve

### 2026-10-20 — phase-1-c CI on GitHub
- Status: done
- Done: user pushed `main` to `S09Z/object-detection-workshop`; `lint-test` workflow ran; switched local work to the `development` branch
- Verify: `gh run list --limit 1` → completed success (ci, main, push, 53s)
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: workflow is development branch → push per sub-task → feature branch from origin/development → draft PR to main

### 2026-10-20 — phase-1-d DVC + remote storage
- Status: done
- Done: `make dvc-init` → `.dvc/` + `.dvcignore` committed; remote `storage` = `/Users/ittichaib/dvc-store` recorded in `.dvc/config.local` (not committed, so every machine sets its own path); a smoke file was added, pushed and then removed
- Verify: `dvc remote list` shows `storage (default)`; `dvc push` → "1 file pushed", file present under `~/dvc-store/files/md5/`; `dvc status -c` → in sync
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: NAS path and backup plan are still open (default `~/dvc-store` has no backup). On the RTX 5070 PC run `make dvc-init DVC_REMOTE=<path>` there. `dvc status -c` warns that `models/export` has no version info until the pipeline has run once
