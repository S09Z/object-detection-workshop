# PROGRESS.md

Two parts:
1. **Current state**: a snapshot that is *overwritten* at the end of every sub-task. A new session reads this first.
2. **Session log**: *append-only*, newest entry at the bottom. Never edit old entries.

Decisions (the *why*) go in `DECISIONS.md`. This file records only *what* and *where we are*.

---

## 1. Current state

> Last updated: 2026-10-29 · by: Claude (moved the RTX 5070 PC from WSL2 to native Windows, D-019)

| Field | Value |
|---|---|
| Current phase | phase-3 — Data (on the RTX 5070 PC, native Windows) |
| Current sub-task | **phase-3-f**: own Thai footage + pre-label (deferred, D-016) or phase-3-g: `dvc add` the data |
| Status | `not started` <!-- not started / in progress / blocked / done --> |
| Last done | phase-3-e: duplicates across splits = 0 after dedup in `prepare`; camera-level leak recorded (`docs/label_issues.md`) |
| Next up | phase-3-g (`dvc add data/raw` + processed) → phase 4 (baseline, parity test, overfit one batch); phase-3-f when footage exists (it would close the tuk_tuk / songthaew gap) |
| Branch / last commit | development (workflow: commit + push per sub-task; feature branch from origin/development → draft PR to main) |
| Blockers | none. Open draft PRs: #7 (iscrowd), #8 (MIO-TCD), #9 (phase-1-f), #10 (roboflow_thai); #4-#6 merged into main, development not yet rebased |

### Health check (latest `./init.sh`)

| Check | Result | When |
|---|---|---|
| ruff check / format | ✅ pass (project ruff via uv; an older ruff had missed 2 E501 + 1 format diff in convert.py) | 2026-10-20 |
| `./init.sh` (WSL2) | ✅ OK | 2026-10-21 |
| `uv sync --all-groups` | ✅ pass (Mac, Python 3.13 venv) | 2026-10-20 |
| `make gpu-check` (RTX 5070, native Windows) | ✅ OK, torch 2.14.1+cu130, sm_120, bf16 True | 2026-10-29 |
| ruff + pytest `-m "not gpu"` (Mac, development @ f8cffa1) | ✅ clean, 30 passed | 2026-10-26 |

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
| Target device | RTX 5070 12 GB PC, native Windows (driver 617.14), CUDA 13 / torch cu130 (D-019, D-014) |
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

### 2026-10-20 — phase-3-c exploratory `make prepare` (not ticked)
- Status: exploration only; phase-3-c stays unchecked
- Done: ran `make prepare` on what is downloaded. Only `coco` is mapped (visaitech_mixed is deliberately not in `source_mappings`; mio_tcd and roboflow_thai are not downloaded). Output is `data/processed/coco/{train,valid,test}` (527 MB, git-ignored) and `reports/data_stats.json`
- Result: train 2,400 / valid 300 / test 300 images. Train boxes: car 5,589, bus 717, truck 1,225, motorcycle 1,099, bicycle 947; **pickup, van, tuk_tuk, songthaew = 0** in every split. Valid/test are thin for some classes (valid bicycle 68, test bus 82), so per-class metrics will be noisy
- Gaps vs the PLAN criterion (every class ≥ 200 train boxes): pickup, van, tuk_tuk, songthaew (need MIO-TCD for pickup/van, Roboflow Thai for songthaew, own footage for tuk_tuk)
- Other observations: all 3,000 COCO images are split per frame (no camera ids; the new warning fired), which is fine for stills; 60 of 9,577 train boxes have `iscrowd=1` and are kept as normal boxes; 40% of train boxes are smaller than 32×32 px
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: consider dropping `iscrowd=1` boxes in `remap_coco`/`prepare` (not changed here). Re-run `make prepare` after more sources are added; do not train on this split, it has 4 empty classes

### 2026-10-20 — drop iscrowd boxes in remap_coco (small fix)
- Status: done (not a PLAN.md sub-task)
- Done: `remap_coco` skips `iscrowd=1` boxes; test `test_remap_drops_crowd_boxes` (failed before the fix). Follows the observation in the exploratory prepare entry above
- Verify: `pytest -m "not gpu"` 27 passed; `make prepare` coco box counts now exclude the crowd boxes
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: phase-3-c is still unchecked (4 classes have no boxes)

### 2026-10-20 — phase-3-b MIO-TCD subset (partial)
- Status: partial (box not ticked)
- Done: read the MIO-TCD page (CC BY-NC-SA 4.0, citation required, localization tar = 3.74 GB); new download hub `http_tar` (curl with resume + safe tar extract) and `mio_tcd` converter in `data/convert.py` (half the subset favours pickup_truck / work_van, drops degenerate boxes); source `mio_tcd` (`enabled: false`, run with `--only`); licenses.md row updated; converter test
- Verify: `data/raw/mio_tcd` has 5,000 images / 16,941 boxes + SOURCE.json; `pytest -m "not gpu"` 28 passed. After `make prepare` (coco + mio_tcd): train 6,400 / valid 800 / test 800 images; train boxes car 13,994, pickup 2,300, van 463, bus 1,126, truck 1,857, motorcycle 1,201, bicycle 936, **tuk_tuk 0, songthaew 0**. Pickup and van now pass the ≥ 200 train-box bar; tuk_tuk and songthaew do not (Roboflow Thai / own footage)
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: `prepare` now mixes a non-commercial source (MIO-TCD) with CC BY ones in one split, so these weights are learning-only (D-010); a commercial split needs a per-source flag in `thai_traffic.yaml` (not implemented). `data/_downloads/mio_tcd` is a 7.0 GB cache (tar + extracted) and can be deleted. MIO-TCD has no camera ids, so same-camera frames can land in different splits; motorcycle/bicycle counts from it are small (63 / 102 boxes)

### 2026-10-21 — phase-1-f RTX 5070 training machine
- Status: done
- Done: Windows driver 617.14 (>= 580); existing WSL2 Ubuntu 24.04; apt build-essential/make/libgl1/libglib2.0-0, uv 0.13.4; cloned to `~/object-detection-workshop` (WSL home), branch development; `make setup` (uv sync --all-groups from the Mac lock; cu130 wheels installed fine on Linux)
- Verify: `make gpu-check` -> torch 2.14.1+cu130, RTX 5070 11.9 GB sm_120, bf16 supported: True, OK: CUDA kernels run on this GPU; `./init.sh` OK
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: `data/` is not on this machine yet (re-run `vd-download` or rsync from the Mac; ask the user before downloading); run `make dvc-init` here. nvidia-smi lives in /usr/lib/wsl/lib, keep it on PATH when scripting wsl. Peak VRAM / s-per-epoch still unmeasured

### 2026-10-22 — phase-3-b Roboflow Thai (partial) + data restored on the 5070 PC
- Status: partial (box not ticked; clips in `data/samples/` still missing)
- Done: user chose `classification-chickhen/thai-traffic-aud0j` v1 (CC BY 4.0, page read 2026-10-22) and `taxi -> car`; `sources.yaml` slug, `source_mappings` (Car, truck, bus, motorcycle, van, taxi, pickup truck, songtaew, Tuktuk) and `docs/licenses.md` row; the `roboflow` hub now calls the REST export API with urllib + zipfile instead of the `roboflow` SDK (not a dependency; test added); `dvc-init` (`~/dvc-store`); re-ran `vd-download` for visaitech, coco, mio_tcd
- Verify: `data/raw`: visaitech 293 / 1,506, coco 3,000 / 32,152, mio_tcd 5,000 / 16,941, roboflow_thai 210 images / 3,003 boxes. pytest 29 passed, ruff clean. `make prepare`: train 6,568 / valid 821 / test 821; train boxes car 15,293, pickup 2,483, van 587, bus 1,125, truck 1,892, motorcycle 1,895, bicycle 936, **tuk_tuk 18, songthaew 24**
- MLflow run / DVC rev: n/a
- Decisions: none (taxi -> car recorded in thai_traffic.yaml)
- Notes for next session: (1) the Universe page says 93 images but the v1 export has 210 = 70 source frames x 3 Roboflow augmentations (only the `train` subdir is used). With no `camera_id` they split per image, so augmented copies of one frame land in different splits (leakage: tuk_tuk is in valid 11 / test 7). Fix before training: group by the source frame (strip `_jpg.rf.<hash>`) or by clip (`footage-N`) in `roboflow_thai`, or use a version without augmentation. (2) Only ~12 distinct tuk_tuk and ~9 distinct songthaew boxes exist, far below the 200 bar: record as a gap in phase-3-c or find more data. (3) After the WSL VM stopped overnight, 3,050 files in `data/raw/mio_tcd` and `reports/data_stats.json` were all NUL bytes (unflushed writes); rebuilt from the intact `data/_downloads/mio_tcd` cache and ran `sync`. Scan `data/` for all-NUL files after any abrupt WSL shutdown. (4) The Windows editor saved `.env` with CRLF; converted to LF. (5) `make` needs `uv` on PATH (`~/.local/bin`)

### 2026-10-22 — fix augmentation leak in roboflow_thai (small fix)
- Status: done (not a PLAN.md sub-task; follows the note in the previous entry)
- Done: `coco_passthrough` takes `camera_id_regex` and sets `camera_id` on each image; `roboflow_thai` uses `^(.+_mp4)-\d+_jpg` so all 210 images (70 frames x 3 augmentations, one clip) form one group; re-converted and re-ran `make prepare`
- Verify: new test `test_coco_passthrough_sets_camera_id_from_file_name` failed before the fix (TypeError) and passes after; `make prepare` -> train 6,568 / valid 821 / test 821; all roboflow_thai images are in train, so tuk_tuk 36 / songthaew 27 are train-only (valid/test = 0, no leakage); full `pytest -m "not gpu"` and ruff run before commit
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: valid/test now have no tuk_tuk or songthaew, so per-class metrics for them cannot be measured until more data (own footage or other sources) exists; the counts are still ~12 / ~9 distinct objects x 3 augmented copies, so 36 / 27 overstates diversity. `vd-download` does not read `.env`: `set -a; . ./.env; set +a` first

### 2026-10-26 — sync PROGRESS with work done in the other (5070 PC) session
- Status: done (bookkeeping only, not a PLAN.md sub-task)
- Done: `git pull` brought development to f8cffa1 (phase-1-f, Roboflow Thai download, clip-group fix); updated §1 only. PLAN.md already has 1-f ticked and 3-b/3-c unticked, which matches the log; no code changed
- Verify: `ruff check`, `ruff format --check` clean; `pytest -m "not gpu"` 30 passed on the Mac
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: HANDOFF.md still says phase-1-f is pending and lists the older data counts; it is stale. PR state: #1-#6 merged (#1-#3 were redundant), #7-#10 draft. Per-source commercial flag for `prepare` is still not started

### 2026-10-26 — per-source commercial flag in `prepare` (small change, not a PLAN.md sub-task)
- Status: done
- Done: `vd-prepare --commercial-only` (`make prepare-commercial`) keeps only sources with `commercial_ok: true` in `sources.yaml` (single source of truth, path in `thai_traffic.yaml` `paths.sources`; a source missing there counts as non-commercial) and writes `data/processed/coco_commercial` + `reports/data_stats_commercial.json`. The default run is unchanged. No DVC stage change (the commercial split is an on-demand output)
- Verify: tests/test_prepare.py (3 tests: flag parsing, every mapped source exists in sources.yaml, end-to-end split excludes the learning source and the default run still includes it); ruff clean; `pytest -m "not gpu"` 33 passed. Real run on the Mac (coco only, roboflow_thai not downloaded here): train 2,400 / valid 300 / test 300; pickup, van, tuk_tuk, songthaew = 0
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: a model trained on `coco_commercial` still needs the image terms of COCO reviewed (docs/licenses.md); pickup/van come only from MIO-TCD, so the commercial set has no pickup/van until another licensed source exists

### 2026-10-27 — phase-3-d EDA
- Status: done (run on the Mac: `data/processed/coco` there = coco + mio_tcd, 8,000 images / 27,214 boxes; no roboflow_thai, so tuk_tuk / songthaew are absent in this run)
- Done: `notebooks/01_eda_dataset.ipynb` now covers integrity check, class balance per split and source, box sizes, density, luminance (day/night proxy), 120 random images with boxes in 4 contact sheets, and the rare classes. Outputs are stripped (no frames in git)
- Verify: `uv run jupyter nbconvert --execute --to notebook --stdout notebooks/01_eda_dataset.ipynb > /dev/null` exit 0; I also looked at two of the contact sheets
- MLflow run / DVC rev: n/a
- Decisions: none
- Observations: (1) car is 64% of train boxes, pickup 10.5%, truck 8.5%, bus 5.1%, motorcycle 5.5%, bicycle 4.3%, van 2.1%; tuk_tuk / songthaew 0 here. (2) 35% of all boxes are < 32x32; car is the worst (44% small), bus the best (6%). (3) COCO is consumer photography (close-ups, people, indoor) while MIO-TCD is fixed traffic cameras: a large domain gap to Thai CCTV, and only 2,430 of 8,000 images are COCO. (4) MIO-TCD gives all pickup (2,885 of 2,885) and van (591) but almost no motorcycle (63) or bicycle (102); COCO gives motorcycle/bicycle. So class and domain are confounded: the model can learn "source" instead of "class". (5) Only 3.5% of sampled images are dark (mean luminance < 70) and none are COCO-night-like; MIO-TCD has some night, rain and snow frames; Thai tropical rain/night glare is not represented. (6) 48 images have no box, boxes/image mean 3.4, p95 10, max 29, far below Bangkok intersection density. (7) Label oddities: COCO "truck" includes fire engines and vans; some MIO-TCD night frames have one huge box over a dark blob; COCO has many non-road scenes (people, kitchen, sheep). (8) **4 raw COCO images are truncated** (65,536 bytes each: 000000463211, 436685, 427401, 159465), likely from the interrupted first download; they are listed in train and test. (9) **`prepare` does not clear its output**: `data/processed/coco/train` has 6,861 files but 6,400 listed images, and some files exist in two splits (stale output of earlier runs). RF-DETR reads the annotation file so training is unaffected, but disk contents are misleading
- Notes for next session: fix 8 and 9 as separate small changes (re-download those COCO images or filter unreadable ones in `prepare`; clear `out` at the start of `prepare`); both need a user decision since re-download is a new download. Re-run the notebook on the 5070 PC after `make prepare` so Thai classes appear. phase-3-e should also check the same-frame duplicates across splits

### 2026-10-27 — prepare: skip unreadable images, clear output dir (small fix)
- Status: done (not a PLAN.md sub-task; follows observations 8 and 9 of the phase-3-d entry)
- Done: `vd-prepare` now drops images that do not fully decode (with their boxes, one `[warn]` line each) and removes `data/processed/coco[_commercial]` before writing, so stale files from earlier runs cannot linger
- Verify: `test_unreadable_image_is_skipped_and_stale_output_is_cleared` fails without the change and passes with it; ruff clean; `pytest -m "not gpu"` 34 passed. Real run on the Mac: the 4 truncated COCO images were skipped; train 6,397 / valid 800 / test 799 images, and file counts on disk equal the annotation counts (was 6,861 / 1,076 / 1,070)
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: the 4 corrupt raw files (65,536 bytes each) are still in `data/raw/coco/images`; re-downloading COCO would replace them (needs the user's OK). Also fixed by accident of the clean-up: no cross-split duplicate files remain
### 2026-10-28 — phase-3-b done: traffic clips in data/samples
- Status: done
- Done: 9 hand-downloaded Pixabay clips (Bangkok, elevated 1080p) in `data/samples/`, 305.0 s = 5.08 min, with `data/samples/SOURCES.md` (URL, length, format per clip) and a row in `docs/licenses.md`; frames of every clip were reviewed, 5 clips rejected for close-up people, moving camera or night blur (30953, 31958, 31964, 31967, 30945; the last one is kept outside the repo in `~/rejected-clips/`)
- Verify: `data/raw/{visaitech_mixed,coco,mio_tcd,roboflow_thai}/annotations.coco.json` have 293 / 3,000 / 5,000 / 210 images, each has `SOURCE.json`, each has a row in `docs/licenses.md` with original and mirror license; clips total 305.0 s (>= 300 s)
- MLflow run / DVC rev: n/a (`dvc add` is phase-3-g)
- Decisions: none
- Notes for next session: Pixabay's Terms of Service prohibit data mining for machine learning purposes and bulk copying, so the clips are **bench / demo only**: never train, label or publish with them (chosen by the user on 2026-10-28, who accepts the risk). Pixabay's CDN refuses scripted downloads, so the clips were fetched by hand. 4 of the 9 clips have a slowly moving camera, so a count accuracy check against manual counts (phase-8) still needs fixed-camera footage, ideally the user's own. Only ~1 clip shows a tuk-tuk (30949). Next: phase-3-c is not ticked until tuk_tuk / songthaew have >= 200 train boxes or the gap is recorded

### 2026-10-29 — phase-3-c prepare + group split (done, with a recorded data gap)
- Status: done (the PLAN criterion allows a recorded gap)
- Done: `make prepare` and `make prepare-commercial` on the 5070 PC with coco + mio_tcd + roboflow_thai; no unreadable-image warnings (COCO was re-downloaded here, so the 4 truncated files from the Mac are not present)
- Verify: both runs succeed; files on disk equal the annotation counts in every split. `coco` split: train 6,568 / valid 821 / test 821 images; train boxes car 15,537, pickup 2,519, van 619, bus 1,106, truck 1,907, motorcycle 2,054, bicycle 923, **tuk_tuk 36, songthaew 27**. `coco_commercial` (commercial_ok sources only, learning weights excluded): train 2,568 / valid 321 / test 321; train boxes car 7,062, pickup 231, van 159, bus 716, truck 1,298, motorcycle 1,963, bicycle 906, tuk_tuk 36, songthaew 27
- **GAP (classes with < 200 train boxes):** tuk_tuk 36 and songthaew 27, all from `roboflow_thai` (about 12 and 9 distinct objects x 3 augmented copies, one clip, train-only). valid/test have **0** of both, so their per-class metrics cannot be measured. In the commercial split also van 159, and pickup / van have 0 in valid/test (MIO-TCD is the only source of them and is non-commercial). Do not train a model meant to count tuk-tuks or songthaews on this data
- MLflow run / DVC rev: n/a (`dvc add` is phase-3-g)
- Decisions: none
- Notes for next session: closing the gap needs new data: own footage (phase-3-f, PDPA review first) or another Thai source with a verified license; at least ~200 train boxes per rare class and a few dozen in valid/test, grouped by camera/clip. phase-3-e should also check same-frame duplicates across splits

### 2026-10-29 — phase-3-e label QA (done)
- Status: done
- Done: `vehicle_detect/data/qa.py` (+ `vd-qa`, tests): duplicate = dHash distance <= 4 **and** mean 64x48 thumbnail difference <= 3 grey levels; box checks (tiny, huge, outside the image, duplicate box). `vd-prepare` now calls `drop_leaks`: an image that duplicates one in an earlier split (train, then valid) is dropped from valid/test (prints `[dedup]`). `docs/label_issues.md` is generated by `vd-qa`. `notebooks/02_label_quality_fiftyone.ipynb` calls the module (the FiftyOne app cell is commented out so the notebook runs headless). Test fixtures in `test_prepare.py` now use distinct noise images, because identical blank ones count as duplicates
- Verify: `vd-qa` after `make prepare`: **duplicate pairs across splits = 0** (removed from valid 25, from test 30); 38 tests pass; ruff clean; the notebook executes (`nbconvert --execute` exit 0). `coco` split now train 6,568 / valid 796 / test 791; `coco_commercial` unchanged (321 / 321, no duplicates there)
- Findings: (1) plain dHash flagged 6,578 cross-split pairs, but most are frames of one fixed MIO-TCD camera with a different car (same background); only ~14 pairs (later 55 images with the thumbnail hash) were real duplicates (same timestamp). (2) **Camera-level leak remains: 5,412 same-scene pairs, 286 valid and 306 test images share a camera view with a train image**, all MIO-TCD (no camera id). Valid/test scores on MIO-TCD classes (pickup, van, car) will be optimistic. Needs a camera id (cluster by scene or read the burnt-in timestamp); not done. (3) 77 suspicious boxes: 61 huge (> 90% of the image), 16 tiny (< 16 px area); all 61 huge and 15 of the tiny ones are COCO, 1 tiny is MIO-TCD; 8 huge ones reviewed by eye are correct labels on consumer close-ups (domain problem, not label error). Nothing was edited or removed; the MIO-TCD "huge box over a dark blob" frames from the EDA were not among the flagged ones
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: the camera-level leak should be fixed before phase 5 reports any metric on pickup/van (e.g. a pseudo `camera_id` from clustering the dHash / thumbnails of MIO-TCD, then `group_by: camera_id`); decide with the user whether to do it before or after phase-3-g. Re-run `vd-qa` after every `make prepare`. `vd-qa` flags MAD <= 3 as duplicate, so two empty-road frames from one camera at the same light can be dropped from valid/test too (conservative; train is never touched)

### 2026-10-29 — move the RTX 5070 PC from WSL2 to native Windows (D-019)
- Status: done (setup; not a PLAN.md sub-task, requested by the user)
- Done: `pyproject.toml` adds `win32` to `tool.uv.environments` and routes torch / torchvision to the `cu130` index and `onnxruntime-gpu` to `win32`; `uv.lock` re-resolved (389 packages, adds pywin32 / pywinpty / waitress). `Makefile`: on Windows `SHELL` is Git Bash and `HOME` comes from `USERPROFILE`; `NVIDIA_SMI` points to `C:/Windows/Sysnative/nvidia-smi.exe` because the chocolatey `make` is 32-bit and cannot see `nvidia-smi.exe` in System32. `.gitattributes`: `* text=auto eol=lf` (the Windows working copy had CRLF, which broke `init.sh`; converted in place). `data/raw` and `data/samples` copied from the WSL clone (8,511 files, same count). `make setup` installed pre-commit and the `nbstripout` filter on Windows (before this, a commit from this clone could have put notebook outputs into git). D-013 is Superseded by D-019; `CLAUDE.md` machine line updated
- Verify: `make gpu-check` -> torch 2.14.1+cu130, RTX 5070 11.9 GB sm_120, bf16 supported: True, OK: CUDA kernels run on this GPU. `./init.sh` (Git Bash) -> `OK — environment is clean` (ruff, 38 tests). `make prepare` on Windows reproduces the WSL result exactly: train 6,568 / valid 796 / test 791, dedup valid 25 / test 30
- MLflow run / DVC rev: n/a
- Decisions: D-019 (Accepted, decided by the user)
- Notes for next session: a Jupyter kernel started from this `.venv` locks `pandas` and makes `uv sync` fail with "Access is denied"; stop the kernel first. `uv run` syncs only the `dev` group, so after `uv sync --all-groups` a plain `uv run` may uninstall a couple of data-group packages and reinstall them; use `uv sync --all-groups` before notebook / FiftyOne work. Still unverified on Windows: a real `make train-5070` (peak VRAM, s/epoch, DataLoader workers use spawn), TensorRT, `onnxruntime-gpu` with CUDA 13 (phase-6). The WSL clone (`~/object-detection-workshop` in the Ubuntu distro) is untouched and can be removed by the user when they no longer need it; it also holds `data/processed` and an 8.4 GB `data/_downloads` cache

### 2026-10-29 — first real RF-DETR Small train on the RTX 5070 (native Windows, smoke)
- Status: done (smoke test, not a PLAN.md sub-task; requested by the user)
- Done: `pyproject.toml` now depends on `rfdetr[train,loggers]>=1.11`; before this `.train()` raised "RF-DETR training dependencies are missing" (`pytorch_lightning` was not in `uv.lock`) on every platform. `uv.lock` re-resolved (adds pytorch-lightning 2.6.6, torchmetrics, wandb, tensorboard, roboflow, pycocotools and others); torch stays 2.14.1+cu130. Smoke config = `rfdetr_small_rtx5070.yaml` with `epochs: 2`, `early_stopping: false`, output `runs/smoke_win_5070` (batch 4 x accum 4, bf16, `num_workers: 8`, eval batch 8, unchanged)
- Verify: `uv run vd-train --config <smoke.yaml>` -> exit 0, no OOM, no DataLoader error with `num_workers: 8` under Windows spawn. 810 s wall (about 215 s per epoch incl. val). val mAP50-95 (EMA) 0.538 -> 0.573; test mAP50-95 0.599, mAP50 0.794; train loss 5.70 -> 5.00. `checkpoint_best_total.pth` saved and load-tested
- MLflow run / DVC rev: `8a32ae70366c4fd791884642c9c59aad` (experiment vehicle-detect, run `smoke-win-5070`); `git_dirty=true`, `dvc_data_rev=unknown`, so it does **not** count under D-017
- Decisions: none
- Notes for next session: **peak VRAM is NOT measured** (CLAUDE.md §6 still owes it): the sampler used `Sysnative
vidia-smi.exe`, which a 64-bit Git Bash cannot see (Sysnative works only for 32-bit processes such as the chocolatey `make`; from Git Bash use `C:/Windows/System32/nvidia-smi.exe`). Re-run 1 epoch with a working sampler. The 2-epoch test mAP is high because valid/test share MIO-TCD camera views with train (camera-level leak, see `docs/label_issues.md`); do not read 0.599 as a real generalisation number. `uv sync` printed "Access is denied" on `orjson` (a process holds the .pyd); imports still work
