# PROGRESS.md

Two parts:
1. **Current state**: a snapshot that is *overwritten* at the end of every sub-task. A new session reads this first.
2. **Session log**: *append-only*, newest entry at the bottom. Never edit old entries.

Decisions (the *why*) go in `DECISIONS.md`. This file records only *what* and *where we are*.

---

## 1. Current state

> Last updated: 2026-10-20 · by: Claude (phase-3-b, partial: visaitech + coco)

| Field | Value |
|---|---|
| Current phase | phase-3 — Data (phase-1-f waits for the RTX 5070 PC) |
| Current sub-task | **phase-3-b**: download and convert to COCO (`visaitech_mixed` and `coco` done; MIO-TCD/UA-DETRAC, Roboflow Thai, clips pending) |
| Status | `in progress` <!-- not started / in progress / blocked / done --> |
| Last done | phase-3-a: `docs/licenses.md` (datasets, models, libraries) |
| Next up | phase-3-b → phase-3-c; phase-1-f when on the 5070 PC |
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
- [x] 1 camera, 1080p (confirmed 2026-10-20)
- [x] Retention periods in `docs/privacy.md` confirmed (2026-10-20)
- [ ] Data controller if real footage is collected (phase-3-f)
- [x] Use case and 95% total / 90% per-class count accuracy confirmed (2026-10-20)
- [x] 9 classes and their order frozen (D-003, 2026-10-20)
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

### 2026-10-20 — phase-1-e MLflow tracking
- Status: done
- Done: chose local tracking (`sqlite:///mlflow.db`, set in `.env` / `.env.example`, same default as `vd-train`); `make mlflow` now serves on :5001 (`MLFLOW_PORT` to override) because macOS AirPlay Receiver answers on :5000 with 403; README updated
- Verify: logged run `phase-1-e-smoke` to `vehicle-detect`; `make mlflow` UI returned HTTP 200 and the run was returned by the search API; the smoke run was then deleted
- MLflow run / DVC rev: n/a
- Decisions: none (local vs central server can be revisited if more than one machine needs to share runs)
- Notes for next session: the 5070 PC keeps its own `mlflow.db`; copy or point `MLFLOW_TRACKING_URI` at a shared location if you want runs from both machines in one UI

### 2026-10-20 — phase-2-a use case and KPI
- Status: done
- Done: `docs/problem.md` with User / KPI / ML metric / Acceptance threshold, built from the PLAN.md defaults (intersection line count, ≥ 95% vs 15-minute manual count); assumptions are tagged `ASSUMPTION` in the file. phase-1-f was skipped because it needs the RTX 5070 PC, not because it is unneeded
- Verify: `docs/problem.md` has all four headings (User, KPI, ML metric, Acceptance threshold)
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: the use case and thresholds are defaults, not confirmed by the user; phase-2-b needs camera count and resolution. `README.md` quick start still says the Linux torch index is `cu128`; it is `cu130` (not changed, outside this sub-task)

### 2026-10-20 — phase-2-b target hardware and latency budget
- Status: done
- Done: `docs/problem.md` now confirms the use case (user, 95% total / 90% per class) and has a per-stage budget table for the RTX 5070 PC mapped onto `vd-bench`'s decode/detect/track stages; camera count and resolution are an assumption (1 camera, 1080p@30)
- Verify: budget sums to 15.5 ms ≤ 16.7 ms (decode 3.0, preprocess 1.5, model 8.0, postprocess 0.5, track 1.5, count 1.0); the bench must run with no training on the same GPU
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: budget numbers are targets, to be measured in phase-6-b. phase-6-a will use model-only ≤ 4.0 ms (50% of the 8.0 ms model row). If real use has several cameras, redo the budget

### 2026-10-20 — phase-2-c privacy (PDPA) DPIA-lite
- Status: done
- Done: `docs/privacy.md`: data inventory, data flow, where images must not appear (git / MLflow / logs / notebooks / demo video) with the enforcing mechanism, retention (PROPOSED), CCTV notice, risks; camera setup (1 camera, 1080p) confirmed in `docs/problem.md`
- Verify: the document states that only counts/boxes leave the device and that no images or plates go into git or MLflow (sections "Data flow" and "Where images must not appear")
- MLflow run / DVC rev: n/a
- Decisions: none (follows D-011)
- Notes for next session: retention periods are proposals; the unauthenticated `/detect` endpoint is listed as an open risk for phase-7-a

### 2026-10-20 — phase-2-d class taxonomy (partial)
- Status: partial (box not ticked)
- Done: `docs/labeling_guide.md` with rules per box, a definition table for the 9 classes in `thai_traffic.yaml` order, ambiguous-case rules (pickup vs truck, van vs car, songthaew); no example images yet because no Thai footage exists (D-016)
- Verify: `make test` passes (25 passed, includes `test_every_config_mapping_points_to_a_real_class`); the taxonomy itself still needs the user's yes
- MLflow run / DVC rev: n/a
- Decisions: D-003 stays `Proposed` until the user confirms
- Notes for next session: on confirmation, set D-003 to `Accepted` (status edit only), tick phase-2-d, then phase-3-a. Add example images during phase-3-d/3-f

### 2026-10-20 — phase-2-d class taxonomy frozen
- Status: done
- Done: user confirmed the 9 classes in order (car, pickup, van, bus, truck, motorcycle, bicycle, tuk_tuk, songthaew); D-003 set to Accepted; labeling guide marked frozen
- Verify: `make test` passes (25 passed, includes `test_every_config_mapping_points_to_a_real_class`)
- MLflow run / DVC rev: n/a
- Decisions: D-003 Accepted
- Notes for next session: class order is now fixed; any change breaks every checkpoint and needs a new decision plus user approval

### 2026-10-20 — phase-3-a license register
- Status: done
- Done: `docs/licenses.md` with every dataset in `source_mappings` and `sources.yaml` (license, mirror license, commercial OK?, phase), the models and key libraries, and a verification checklist
- Verify: script check: every key in `source_mappings` (coco, mio_tcd, roboflow_thai) and every source in `sources.yaml` has a row in `docs/licenses.md`
- MLflow run / DVC rev: n/a
- Decisions: none (follows D-010)
- Notes for next session: licenses come from the project config, not from reading the source pages; the `Verified` column is "No" everywhere, so do the checklist at the bottom of the file before `make download` in phase-3-b. `roboflow_thai` workspace/project are still `CHANGE_ME`

### 2026-10-20 — phase-3-b dataset download (partial: visaitech_mixed)
- Status: partial (box not ticked)
- Done: read the dataset card (CC BY 4.0, commercial OK with attribution); pinned `revision` de35185 in `sources.yaml`; `make download` fetched images/labels only (27 MB, skipped the 12 GB videos) and wrote `data/raw/visaitech_mixed/{images,annotations.coco.json,SOURCE.json}`; fixed `camera_id_regex` so the 23 `VEHICLES__*` frames get a camera id (they were silently ungrouped, which would have split them per frame); updated the `visaitech_mixed` row in `docs/licenses.md`
- Verify: partial. `visaitech_mixed`: 293 images, 1,506 boxes (matches the card), SOURCE.json present, licenses.md row has original and mirror license; camera ids V001–V005 + VEHICLES, none missing. The PLAN criterion covers every source (COCO subset, MIO-TCD/UA-DETRAC, Roboflow Thai, ≥ 5 min of clips in `data/samples/`) and those were not downloaded
- MLflow run / DVC rev: n/a (`dvc add data/raw` is phase-3-g)
- Decisions: none
- Notes for next session: only 5 + 1 camera groups here, so a 0.8/0.1/0.1 split will hit the new empty-split guard on this source alone; merge with the other sources before `make prepare`. `visaitech_mixed` is not in `source_mappings` on purpose (its 4-wheeler class mixes car/pickup/van/bus/truck). Next downloads need user permission and the license checklist in `docs/licenses.md`

### 2026-10-20 — phase-3-b COCO vehicle subset (partial)
- Status: partial (box not ticked)
- Done: read COCO terms of use (annotations CC BY 4.0; images under Flickr terms); new download hub `fiftyone_zoo` in `data/download.py` (sets `FIFTYONE_DATASET_ZOO_DIR` to the cache, downloads only images with the requested classes); source `coco` in `sources.yaml` (3,000 images, seed 42); `docs/licenses.md` row updated; test that every source declares `license` and `commercial_ok`
- Verify: `data/raw/coco/{images,annotations.coco.json,SOURCE.json}` present; 3,000 images, 32,152 boxes, 0 missing image files; after `source_mappings`: car 6,892, bus 902, truck 1,553, motorcycle 1,427, bicycle 1,108; `pytest -m "not gpu"` 26 passed. The PLAN criterion still needs MIO-TCD/UA-DETRAC, Roboflow Thai and ≥ 5 min of clips
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: the first attempt failed after 1.3 GB on a DNS error; a re-run resumed from the cache. `data/_downloads/coco` (1.4 GB) is a cache and can be deleted once DVC tracks `data/raw`. COCO images have no camera ids, so they split per frame (fine for stills). pickup, van, tuk_tuk and songthaew have no boxes yet
