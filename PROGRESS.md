# PROGRESS.md

Two parts:
1. **Current state**: a snapshot that is *overwritten* at the end of every sub-task. A new session reads this first.
2. **Session log**: *append-only*, newest entry at the bottom. Never edit old entries.

Decisions (the *why*) go in `DECISIONS.md`. This file records only *what* and *where we are*.

---

## 1. Current state

> Last updated: 2026-11-17 · by: Claude (phase-4-f: checkpoint / resume / lineage drill)

| Field | Value |
|---|---|
| Current phase | phase-3 — Data (on the RTX 5070 PC, native Windows) |
| Current sub-task | **phase-4-e**: counting demo (needs a confident model: use `runs/lineage_5ep/checkpoint_best_total.pth`) |
| Status | `not started` <!-- not started / in progress / blocked / done --> |
| Last done | phase-4-f: 5-epoch run stopped after epoch 2 and resumed from epoch 2; lineage tags, `model/` artifact and archive pruning verified |
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
| test mAP50-95 (5 COCO classes, pretrained, no training; 808-image test split after the leak fix) | > baseline phase-4-a | 0.568 strict / 0.565 merged | `reports/baseline_pretrained.json` (git-ignored), code `src/vehicle_detect/baseline.py` |
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

### 2026-10-29 — RTX 5070 peak VRAM and s/epoch (1-epoch smoke, native Windows)
- Status: done (measurement; closes the VRAM item left open by the entry above)
- Done: re-ran `rfdetr_small_rtx5070.yaml` with `epochs: 1`, output `runs/smoke_win_5070_1ep`, sampling `nvidia-smi` every 2 s (`C:/Windows/System32/nvidia-smi.exe` from Git Bash)
- Verify: `uv run vd-train --config <smoke1.yaml>` -> exit 0. Peak GPU memory **6,398 MiB** of 12,227 (about 4.7 GB above the ~1.7 GB desktop baseline; nvidia-smi counts PyTorch's cached allocator pool, so the true need is lower). Batch 4 x accum 4, bf16, no OOM, so there is headroom for a larger physical batch. Wall 633 s for the whole command (about 9 min 30 s from first to last log line: model load, 1 train epoch, val, test). Epoch 1 val mAP50-95 (EMA) 0.537. GPU utilisation peaked at 82% but averaged only about 38% during the train window (22% over the whole run), so the GPU is partly starved; likely dataloader / CPU bound (`num_workers: 8`, 16 logical cores, Windows spawn). Not yet diagnosed
- MLflow run / DVC rev: run `smoke-win-5070-1ep` (experiment vehicle-detect); `git_dirty=true` and no DVC data rev, so it does not count under D-017
- Decisions: none
- Notes for next session: before the real 50-epoch run, try `batch_size: 8` + `grad_accum_steps: 2` (same effective 16) and compare s/epoch and utilisation as one change; if utilisation stays low, look at dataloader workers / image decode

### 2026-11-09 — why GPU utilisation was low: worker start-up on Windows (1-epoch probes)
- Status: done (diagnosis; follows the 2026-10-29 peak-VRAM entry, which left the low utilisation undiagnosed)
- Done: 1-epoch probes of `rfdetr_small_rtx5070.yaml` on a fixed 960-image subset of `data/processed/coco` (seed 0; 64 valid, `run_test: false`, hard-linked into git-ignored `runs/bench_data`), batch 4 x accum 4, changing only `num_workers`. A game was holding the GPU during the first probes; it was closed for the rows below and the result did not change
- Verify: time of the train phase (log line "Building ... train dataset" to "Skipping validation"): `num_workers` 8 -> 293 / 313 / 319 s (3 runs), 4 -> 144 s, 2 -> 98 s, 0 -> 52 s. Peak GPU memory 5.9-6.1 GB in all cases. Batch 8 x accum 2 with 8 workers -> 274 s (not significant vs. 293-319 s). With 8 workers GPU util averages 6% and CPU 22%, so neither is the limit: the time is a start-up cost that grows with the worker count (about 25-35 s per worker, consistent with Windows `spawn` re-importing torch/rfdetr in each worker)
- MLflow run / DVC rev: runs `bench-*` (experiment vehicle-detect); `git_dirty=true`, do not count under D-017
- Decisions: none
- Notes for next session: this measures start-up, not steady state. `persistent_workers` is on by default, so the cost is paid once per run, not per epoch (the 2-epoch run's 2nd epoch took ~215 s with 8 workers incl. val). The earlier "38% average GPU utilisation" came from a 1-epoch run that included this start-up, so it overstated the problem. Do not change `num_workers` in the config yet: first compare steady-state epoch time (3-epoch run, epochs 2-3) for `num_workers` 8 vs 2 vs 0, since 0 moves decoding into the main process and may be slower per epoch

### 2026-11-10 — steady-state epoch time vs num_workers (3-epoch probes, 960-image subset)
- Status: done (measurement; follows the 2026-11-09 start-up entry)
- Done: same subset and config as the 2026-11-09 probes, `epochs: 3`, only `num_workers` changed (8, 2, 0). Steady-state epoch time = gap between the "Best EMA metric improved" log lines of consecutive epochs (train + val of 64 images)
- Verify: `num_workers` 8 -> first epoch done after 240 s, then 33 s per epoch; 2 -> first epoch after 85 s, then 31 s per epoch; 0 -> first epoch after ~41 s, then 40 s per epoch. Peak GPU memory 5.8-5.9 GB. Mean GPU util over the whole run 13.5% (8) / 29% (2) / 34% (0); mean CPU 23% / 36% / 46%. So the per-epoch speed is the same for 8 and 2 workers (about 30 images/s, matching the full-data 2-epoch run), and 0 workers is ~25% slower per epoch but has no start-up cost
- MLflow run / DVC rev: runs `bench-ep3_*`; `git_dirty=true`, do not count under D-017
- Decisions: none
- Notes for next session: the ~2.5 min saved by 2 workers at start is ~1.5% of a 50-epoch run, so `num_workers: 8` in `rfdetr_small_rtx5070.yaml` was left unchanged; use `num_workers: 2` for short probes and smoke runs. One run per setting, subset only, so differences under ~10% are within noise. Low GPU utilisation is therefore not a data-loading problem; the 5070 is simply not saturated at batch 4 (peak 6 GB of 12): the next single change to try is `batch_size: 8` + `grad_accum_steps: 2` over a full epoch

### 2026-11-10 — batch 8 x accum 2 vs batch 4 x accum 4, one full epoch (6,568 images, 8 workers)
- Status: done (measurement; single change vs. the config, as in the 2026-11-09/10 probes)
- Done: `rfdetr_small_rtx5070.yaml` with `epochs: 1`, `run_test: false`, full `data/processed/coco`, only batch/accum changed; GPU idle beforehand (no game); both runs back to back
- Verify: train phase (incl. ~4 min worker start-up with 8 workers) 414 s for batch 8 x 2 vs 435 s for batch 4 x 4 (-5%, within noise); whole command 509 s vs 484 s. Peak GPU memory **10,036 MiB vs 6,153 MiB** (of 12,227). Mean GPU util 25% vs 27%. Epoch-1 val EMA mAP50-95 0.532 vs 0.527, train loss 5.79 vs 5.73 (same effective batch 16, so no difference expected)
- MLflow run / DVC rev: runs `bench-full_b8a2`, `bench-full_b4a4`; `git_dirty=true`, do not count under D-017
- Decisions: none
- Notes for next session: batch 8 gives no speed-up but leaves only ~2 GB free, so `rfdetr_small_rtx5070.yaml` stays at batch 4 x accum 4. Mean GPU util is 25-27% in both because the 4 min start-up is included; the GPU is not the limit of a 1-epoch run, and for the 50-epoch run the start-up is amortised. Next lever, if epoch time matters, is not batch size (try resolution / `amp` only with an MLflow-compared run)

### 2026-11-13 — phase-3-g: version data with DVC
- Status: done
- Done: `dvc add data/raw` -> `data/raw.dvc` (md5 `a3d359377f43a482856162129440d914.dir`, 8,511 files, 720,825,243 bytes); `dvc push` to the local remote `C:/Users/tommy/dvc-store` (8,481 unique objects, 707 MB). `.gitignore` changed from `/data/` to `/data/*` + `!/data/*.dvc` + `!/data/.gitignore`: DVC refuses a `.dvc` file that git ignores, and the contents of `data/` stay ignored. The remote is machine-local (`.dvc/config.local`, `make dvc-init` equivalent run by hand with the Windows path, `uv run --no-sync dvc remote add -d storage C:/Users/tommy/dvc-store --local`). `data/processed` and `data/samples` are not tracked (they are derived / bench-demo only)
- Verify: `uv run dvc status -c data/raw.dvc` -> "Cache and remote 'storage' are in sync."; `dvc status data/raw.dvc` -> "Data and pipelines are up to date."; `git log -- data/raw.dvc` shows this commit. The first plain `dvc push` printed "Everything is up to date" while the remote was still empty (0 bytes); a second `dvc push -v` transferred 8,481 files, which is what the numbers above are from
- MLflow run / DVC rev: n/a; DVC data rev for lineage (D-017) = `data/raw.dvc` at this commit
- Decisions: none
- Notes for next session: `uv sync --all-groups` then `uv run --no-sync ...` (plain `uv run` syncs only `dev` and may remove `dvc`); `orjson` had to be reinstalled (`--reinstall-package orjson`) after the earlier "Access is denied". The camera-level leak (MIO-TCD views shared between train and valid/test) is a `prepare` / split change, so it does not alter `data/raw` and can still be fixed after this; it would need a new sub-task, then a train whose lineage counts under D-017 (clean tree + `dvc_data_rev`)

### 2026-11-16 — phase-4-a: COCO-pretrained baseline (no training)
- Status: done
- Done: `src/vehicle_detect/baseline.py` (+ `tests/test_baseline.py`) runs the stock RF-DETR Small (COCO weights, 90 classes, threshold 0.01) on `data/processed/coco/test` (791 images, 2,632 boxes) and maps COCO bicycle/car/motorcycle/bus/truck onto our taxonomy. pickup, van, tuk_tuk and songthaew have no COCO class. Run: `uv run --no-sync python -m vehicle_detect.baseline`. Two ground-truth views: `strict` drops pickup/van boxes (a COCO "car" on a pickup is a false positive), `merged` relabels pickup/van as car
- Verify: mAP50-95 per COCO-known class (strict): car 0.503, bus 0.796, truck 0.291, motorcycle 0.566, bicycle 0.480; overall 0.527 (mAP50 0.722). merged: car 0.519, rest identical, overall 0.530 (mAP50 0.728). `ruff check` / `ruff format --check` clean, `pytest` all pass
- MLflow run / DVC rev: n/a (inference only; the JSON is in git-ignored `reports/`)
- Decisions: none
- Notes for next session: this is the bar a trained model must clear on the same 5 classes; the 2-epoch smoke fine-tune scored test mAP50-95 0.599 over 7 classes, which is not comparable (different class set) and is inflated by the camera-level leak. Test GT has 287 pickup + 59 van boxes that the baseline cannot name, so the real 9-class gap is larger than 0.527 suggests; tuk_tuk / songthaew have no test boxes at all (phase-3-c gap). `truck` is the weakest class (0.29): COCO "truck" vs our truck/pickup/van boundary. Next: phase-4-b (ONNX parity test, needs an ONNX export of the model)

### 2026-11-17 — phase-4-b: ONNX parity test
- Status: done
- Done: exported the 2-epoch smoke checkpoint to ONNX (fp32, batch 1; outputs `dets` 1x300x4 and `labels` 1x300x10 = 9 classes + background slot, as `postprocess` assumes) and compared `OnnxDetector` with `RFDETR.predict()` on 10 test images (every 79th, conf >= 0.5, 28 reference detections). First run: 27/28 matched, 1 miss at IoU 0.9875 on a tiny box touching the left edge: the ONNX box had x1 = -0.4 while `predict()` clips to 0. Everything else agreed within 0.03 px. Fix: `postprocess` clips boxes to the image. Float-before-resize in `preprocess` was tried and did not change the miss, so `preprocess` is unchanged
- Verify: `uv run pytest -m slow tests/test_parity.py` -> passed (every reference detection has an ONNX box with IoU >= 0.99 and the same class; equal detection counts per image). `pytest -m "not slow and not gpu"` -> all pass (41), `ruff check` / `ruff format --check` clean. New CPU test `test_postprocess_clips_boxes_to_the_image`
- MLflow run / DVC rev: n/a
- Decisions: none
- Notes for next session: `tests/test_parity.py` is marked `slow` (marker added to `pyproject.toml`); it skips when no `runs/*/checkpoint_best_total.pth` or no `data/processed/coco/test` exists, and `PARITY_CHECKPOINT` picks a checkpoint. It ran on the CUDA execution provider, not CPU. On native Windows the TensorRT provider fails to load (`nvinfer_10.dll` missing) and ONNX Runtime falls back to CUDA: TensorRT must be installed before phase-6 (D-019 risk list). The checkpoint used was the 2-epoch smoke one, which is enough for parity but says nothing about accuracy

### 2026-11-17 — camera-level leak: pseudo camera ids for MIO-TCD, baseline re-measured (follow-up to phase-3-e)
- Status: done (not a PLAN.md sub-task; the leak was recorded in phase-3-e and is a precondition for a train that counts under D-017)
- Done: `qa.assign_scene_ids` gives every MIO-TCD image a pseudo `camera_id` before `group_split`, enabled by `split.scene_cluster_sources: [mio_tcd]` in `configs/data/thai_traffic.yaml`; `vd-prepare` calls it and prints `[scenes] N pseudo camera ids`. Two metrics were tried and rejected on visual evidence: (1) dHash <= 6 single-link (1,192 groups) left real same-camera pairs across splits: 8 of 8 sampled pairs at dHash 7-8 were one camera (a large vehicle changes enough hash bits); (2) 70% pixel match on a 32x24 thumbnail (399 groups) still missed the same camera at another hour or at night. Final metric: sign of (64x48 grayscale - Gaussian blur) where contrast exists, score = (agreeing - disagreeing) / (pixels with an edge in either) >= 0.4, single-link. On the 5,000 MIO images: 208 groups, largest 403 (8%); 0.5 gives 332 groups with the same largest group, 0.3 chains into one 3,488-image group, so 0.4 sits on a stable plateau. 8 of 8 pairs joined only between 0.4 and 0.5 were one camera (incl. night frames)
- Verify: `make prepare`-equivalent (`uv run --no-sync vd-prepare`) -> train 6,581 / valid 821 / test 808 images (was 6,568 / 796 / 791). `vd-qa`: duplicate pairs across splits 0; same-scene pairs (dHash <= 4) across splits **2** (was 5,412; valid 1 image, test 0, was 286 and 306), the 2 being a COCO still vs MIO frames (different datasets). At the looser dHash <= 8: 207 pairs (was 707 after the first attempt), but 8 of 8 sampled remaining mio-mio pairs are different cameras, so they are dHash false positives. Class counts after the split: valid and test still have all 7 classes with data, tuk_tuk 36 / songthaew 27 remain train-only. `pytest -m "not slow and not gpu"` all pass (42); ruff clean on the touched files. Baseline re-run on the 808-image test split: strict mAP50-95 0.568 (car 0.499, bus 0.793, truck 0.420, motorcycle 0.605, bicycle 0.522), merged 0.565
- MLflow run / DVC rev: n/a. `data/processed` is not DVC-tracked (derived; `data/raw` is, phase-3-g)
- Decisions: none
- Notes for next session: the baseline moved from 0.527 to 0.568 only because the test images changed (truck 0.29 -> 0.42), so any comparison must name the split it was measured on; the 2-epoch smoke runs and their 0.599 were trained and tested on the old, leaky split and are obsolete. The leak is removed by construction for the MIO-TCD view metric, but the metric cannot see a camera that changes its framing; it is checked by sampling (16 pairs by eye), not proven. Roboflow (clip ids) and visaitech (video ids) already had camera ids; COCO stills are not from fixed cameras. `docs/label_issues.md` was regenerated by `vd-qa`. The scene features are recomputed on every prepare (about 35 s for 5,000 images)

### 2026-11-17 — phase-4-c: overfit one batch
- Status: done
- Done: `vehicle_detect.data.subset` (new, tested; also for phase-4-d) writes 16 random train images (seed 0, 53 boxes: car 27, bus 14, pickup 6, bicycle 2, truck 4) as train, valid and test (`--same`) into `data/processed/overfit16`. `configs/train/overfit16.yaml`: RF-DETR Small, batch 4 x accum 1 (4 optimizer steps per epoch), lr 1e-4, 100 epochs, no early stopping, `num_workers: 0`, bf16, default augmentation, run name `overfit-16`. Run: `uv run vd-train --config configs/train/overfit16.yaml`
- Verify: exit 0; val mAP50 on the same 16 images: 0.052 (epoch 0), 0.839 (10), **0.987 (20)**, 0.997 (40), 1.000 from epoch 60 to 99, for both the regular and the EMA weights (criterion: >= 0.95). Train loss 8.65 (epoch 0) -> 2.26 (epoch 99): it is still falling and is not near 0, so the PLAN goal wording "loss near 0" is not literally met; the loss includes the auxiliary decoder losses and is computed on augmented images, while the mAP is computed on the clean images. The Verify criterion (mAP50 >= 0.95) is the one used. `pytest -m "not slow and not gpu"` passes, ruff clean on the new files
- MLflow run / DVC rev: `overfit-16` (id `eb830f9ccf90403c8e7aa8f066e82a8e`, experiment vehicle-detect); tags `git_sha=82a8db1`, `git_dirty=true`, `dvc_data_rev=unknown`, so it does not count under D-017 (a throw-away sanity run); checkpoint not logged as an artifact
- Decisions: none
- Notes for next session: the pipeline can fit labelled data, so the labels, class mapping and COCO conversion are consistent with the model's outputs; this does not prove generalisation. If phase-4-d or later shows a flat loss, the data path is not the first suspect any more. `runs/overfit16` and `data/processed/overfit16` are git-ignored

### 2026-11-17 — phase-4-d: smoke run of the whole pipeline
- Status: done (with one deviation from the PLAN wording, below)
- Done: `pipelines/smoke/dvc.yaml` (own `dvc.lock`, tracked): `subset` (500 train / 100 valid / 100 test from `data/processed/coco`, `vehicle_detect.data.subset`) -> `train` (`configs/train/smoke.yaml`, 2 epochs, `num_workers: 2`) -> `export` (`configs/export/smoke.yaml`, ONNX fp32) -> `evaluate` -> `bench` (`configs/bench/smoke.yaml`, 200 frames of one Pixabay clip, gate disabled: `target_fps: 0`). All outputs go to `runs/smoke`, `models/smoke_export`, `reports/smoke/`, `data/processed/smoke`, so the real pipeline paths are untouched. `make repro-smoke` = `uv run --no-sync dvc repro pipelines/smoke/dvc.yaml` (`--no-sync`: a plain `uv run` syncs only the dev group and would uninstall dvc mid-run). `evaluate.py` reads an optional `eval_report_path`. Two real defects found by the run and fixed: (1) `OnnxDetector` could not use the CUDA provider in a process that had not imported torch (`cudnn64_9.dll` LoadLibrary failed), so `vd-evaluate`, `vd-bench`, `vd-count` all failed on Windows; it now calls `onnxruntime.preload_dlls()` when the CUDA provider is present; (2) `configs/bench/realtime.yaml` pointed at `models/export/inference_model.onnx` but `vd-export` writes `rfdetr-small.onnx`
- Verify: `make repro-smoke` -> exit 0 (first attempt failed at `evaluate` on the cuDNN defect; after the fix `dvc repro` skipped the unchanged `subset`/`train`/`export` and ran `evaluate` and `bench`). `reports/smoke/eval.json`: mAP50-95 0.235, mAP50 0.339 (100 test images, 2 epochs, so only a plumbing check). `reports/smoke/bench.json`: 200 frames, 66.7 FPS, p50 14.3 ms, p99 20.1 ms (decode 1.8, detect 13.1, track 0.05 ms), provider CUDAExecutionProvider (fp32, no TensorRT), so this is NOT the 60 FPS gate result: the gate is off in this config. `ruff check src tests`, `ruff format --check`, `pytest -m "not slow and not gpu"` all pass
- Deviation: the PLAN Verify says `uv run dvc repro`; running that on the real `dvc.yaml` would re-run `prepare`, train 50 epochs on the full set and overwrite `data/processed/coco`, `runs/rfdetr_small_rtx5070` and `models/export`, which contradicts the Goal ("on a small subset, 2 epochs"). The same chain was run through the separate smoke pipeline instead, and the reports are `reports/smoke/eval.json` and `reports/smoke/bench.json` rather than `reports/eval.json` / `reports/bench.json` (DVC refuses two stages declaring one output). Say so if the literal paths matter
- MLflow run / DVC rev: run `smoke-pipeline` (experiment vehicle-detect), `git_dirty=true`, does not count under D-017
- Decisions: none
- Notes for next session: the smoke `subset` stage deliberately has no dependency on `data/processed/coco` (it is an output of the real `prepare` stage and would make dvc re-run it); after a new `make prepare` run it with `dvc repro -f`. First end-to-end timing on the RTX 5070 with CUDA EP fp32: 66.7 FPS / p99 20.1 ms including a tracker; the p99 misses the 16.7 ms budget, TensorRT fp16 (phase-6) is the lever. The TensorRT provider still cannot load on this machine (`nvinfer_10.dll`)

### 2026-11-17 — phase-4-f: checkpoint, resume and lineage (D-017)
- Status: done (executed before phase-4-e on purpose: the counting demo needs a model confident enough for ByteTrack, and this run produces one)
- Done: `configs/train/lineage5.yaml` (RF-DETR Small, full `data/processed/coco`, 5 epochs, `checkpoint_interval: 1`, `keep_last_checkpoints: 2`, `num_workers: 2`, `run_test: true`). Prepared the lineage inputs first: `UV_NO_SYNC=1 uv run --no-sync dvc repro prepare` wrote `dvc.lock` (`data/processed/coco` md5 `575ffd68829193b16982658170d15a3e.dir`, 685 MB), committed; cleared 47 stat-dirty tracked files with `git add -u` (no content change) so `git_dirty` could read false. Run 1 was stopped after epoch 2 (process tree killed once `checkpoint_1.ckpt` was complete), then `make train-resume TRAIN_CONFIG=configs/train/lineage5.yaml`. Defect found and fixed (commit `f9b0ca6`): RF-DETR writes **no `last.ckpt` when `checkpoint_interval == 1`**, so `--resume auto` found nothing and would have restarted at epoch 0 without a warning; `checkpoints.resolve_resume("auto", ...)` now falls back to the newest `checkpoint_<n>.ckpt` (unit-tested)
- Verify: (1) the resumed run printed `Resuming from runs\lineage_5ep\checkpoint_1.ckpt` and `metrics.csv` continues at epochs 2, 3, 4 (not 0): val EMA mAP50-95 0.500, 0.521 (run 1), 0.538, 0.541, 0.539 (resumed run); test mAP50-95 0.546, mAP50 0.733 (808 images, 7 classes). (2) `checkpoint_best_total.pth` loads (`vd-train` load-tests it after training; exit 0). (3) interval archives left: `checkpoint_3.ckpt`, `checkpoint_4.ckpt` = `keep_last_checkpoints` (0, 1, 2 were pruned); best / last_ema kept. (4) MLflow run `lineage-5ep-resume` (id `5ff03767...`, FINISHED): tags `git_sha=f9b0ca61802f...`, `git_dirty=false`, `dvc_data_rev=575ffd688291...`, `sha256=9313d1ca90f7...`, `resumed_from=runs\lineage_5ep\checkpoint_1.ckpt`; artifact `model/` holds `checkpoint_best_total.pth`, `model_meta.json`, `train_config.yaml`, `data_config.yaml`. `pytest -m "not slow and not gpu"` passes, ruff clean
- MLflow run / DVC rev: `lineage-5ep-resume`, `git_sha=f9b0ca6`, `dvc_data_rev=575ffd68829193b16982658170d15a3e.dir`, `git_dirty=false`: this run **counts** under D-017. The interrupted first run `lineage-5ep` (98e46006) carries no tags (it never reached the lineage step) and is marked KILLED
- Decisions: none
- Notes for next session: speed on the RTX 5070 with 2 workers, batch 4 x 4: about 3.4 min per epoch including validation, 13.5 min for the 3 resumed epochs plus test. A tracked-file `git status` that is not empty (even CRLF-only) makes `git_dirty=true`: run `git add -u` first, and do not edit tracked files while a run is going. A run started from a different commit gets a different `git_sha`: the resumed run's `git_sha` is the commit of the resume, not of epoch 0-1. The model is a 5-epoch result, not the phase-5-a baseline
