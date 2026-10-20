# DECISIONS.md

A log of technical decisions (lightweight ADRs). Explains **why** we chose something, so a later
session doesn't reverse it without a reason or repeat a discussion that has already happened.

## Rules for agents

- **Add a record** whenever you choose between options that affect later work: model, dataset,
  tool, architecture, metric, threshold, or a deliberate deviation from `PLAN.md`.
  A routine choice that is easy to change back does not need a record.
- **Append-only.** Never edit the content of an old decision. When something changes, add a new
  record and change the old one's status to `Superseded by D-00N` (the only edit allowed).
- Decisions marked **needs user approval** in `CLAUDE.md` §5 (license, privacy, taxonomy,
  framework switch, changing a Verify criterion) must be recorded as `Proposed`, and become
  `Accepted` only after the user agrees.
- Keep each record short: 5–15 lines. If more detail is needed, put it in `docs/` and link to it.
- Reference the ID (`D-00N`) in commit messages and in `PROGRESS.md` when it relates.

### Template

```
## D-00N — <short title>
- Date: YYYY-MM-DD · Phase: phase-X-y · Status: Proposed | Accepted | Superseded by D-00M | Rejected
- Decided by: user | agent (approved by user) | agent
- Context: the problem or constraint, in 1–3 lines
- Options: A) … B) … C) …
- Decision: the option chosen
- Why: main reasons, with numbers or sources where possible
- Consequences: what follows, and what it costs
- Revisit if: the condition that should trigger re-deciding
```

---

## Index

| ID | Title | Status |
|---|---|---|
| D-001 | Primary detector: RF-DETR (Apache-2.0); YOLO26 is optional only | Accepted |
| D-002 | The real-time metric is end-to-end FPS + p99 on the target device | Accepted |
| D-003 | 9-class Thai traffic taxonomy, fixed order | Proposed (user to confirm in phase-2-d) |
| D-004 | Split data by camera/video, not by frame | Accepted |
| D-005 | Inference via ONNX Runtime / TensorRT, independent of PyTorch | Accepted |
| D-006 | FP16 by default; INT8 only if FP16 doesn't pass | Accepted |
| D-007 | ByteTrack + supervision LineZone for counting | Accepted |
| D-008 | Tool stack: uv, ruff, DVC, MLflow, FiftyOne, pre-commit | Accepted |
| D-009 | Split evidently and fiftyone into separate environments | Accepted |
| D-010 | Public datasets for learning; production uses COCO/CC BY + in-house Thai data | Accepted |
| D-011 | Privacy by design: send only counts/boxes out of the device | Accepted |
| D-012 | Harness: PLAN.md (Verify) + PROGRESS.md + DECISIONS.md instead of a JSON feature list | Accepted |
| D-013 | Train on RTX 5070 12 GB with PyTorch cu130, via WSL2 | Accepted |
| D-014 | Target device = RTX 5070 PC (edge server) | Accepted |
| D-015 | DVC remote = local disk / NAS (on-premise) | Accepted |
| D-016 | No Thai footage yet: use public data + freely licensed clips first | Accepted |
| D-017 | Checkpoint & run lineage: best by val mAP, resume, seed, test, MLflow artifacts + registry | Accepted |
| D-018 | Pull datasets from HF / Kaggle / Roboflow with `vd-download`, separate from training inclusion | Accepted |

---

## D-001 — Primary detector: RF-DETR (Apache-2.0); YOLO26 is optional only
- Date: 2026-10-20 · Phase: setup · Status: Accepted
- Decided by: agent (from the workshop research)
- Context: the model must exceed 60 FPS and be usable in a commercial product
- Options: A) Ultralytics YOLO26 B) RF-DETR Nano/Small/Medium C) D-FINE / DEIMv2
- Decision: B as the default, with A as an optional extra (`--extra yolo`) for comparison
- Why: on a T4, RF-DETR-N gets 48.4 mAP at 2.3 ms vs YOLO26-N at 40.3 mAP at 1.7 ms. Both are far under the 16.7 ms budget.
  RF-DETR Nano/Small/Medium are Apache-2.0. Ultralytics states that AGPL-3.0 also covers fine-tuned weights.
  RF-DETR also leads on RF100-VL (fine-tuning on new domains). DEIMv2 has a license that requires contacting the vendor for commercial use.
- Consequences: Nano has 30.5M parameters (YOLO26n has 2.4M), which uses more memory on small Jetson boards. No Jetson benchmark for RF-DETR has been published yet.
- Revisit if: RF-DETR doesn't reach 60 FPS end-to-end on the device in phase-6-c, or a YOLO Enterprise license is bought

## D-002 — The real-time metric is end-to-end FPS + p99 on the target device
- Date: 2026-10-20 · Phase: setup · Status: Accepted
- Decided by: agent
- Context: published model latencies leave out decode, pre/post-processing and tracking
- Decision: pass = mean FPS ≥ 60 **and** p99 ≤ 16.7 ms over ≥ 5 minutes of video, after warm-up, measured by `vd-bench`
- Why: one Jetson Orin Nano case ran about 90 FPS for the engine alone, but 33 FPS in a C++ app and 16 FPS through Python ORT. Mean FPS alone hides stutter.
- Consequences: `vd-bench` exits with code 1 when it fails, so CI blocks the release. Results from a Mac do not count.
- Revisit if: the use case changes to processing that isn't real-time (e.g. batch overnight)

## D-003 — 9-class Thai traffic taxonomy, fixed order
- Date: 2026-10-20 · Phase: setup → phase-2-d · Status: Proposed
- Decided by: agent (needs user approval)
- Context: the 5 COCO vehicle classes don't separate pickup, songthaew or tuk-tuk, which are common in Thailand
- Decision: car, pickup, van, bus, truck, motorcycle, bicycle, tuk_tuk, songthaew, in this order (order = class_id)
- Consequences: changing the order or the number of classes makes every existing checkpoint unusable, so it requires user approval
- Revisit if: the business KPI from phase-2-a needs more or fewer classes

## D-004 — Split data by camera/video, not by frame
- Date: 2026-10-20 · Phase: setup · Status: Accepted
- Decided by: agent
- Context: consecutive frames from one camera look almost identical
- Decision: `group_split(group_by="camera_id")`; an image without a `camera_id` is treated as its own group
- Why: splitting by frame leaks near-duplicate images into valid/test, so mAP comes out higher than reality
- Consequences: each source dataset needs a `camera_id`/`video_id` added during conversion (phase-3-b)

## D-005 — Inference via ONNX Runtime / TensorRT, independent of PyTorch
- Date: 2026-10-20 · Phase: setup · Status: Accepted
- Decided by: agent
- Decision: `OnnxDetector` does its own pre/post-processing in numpy and picks the provider TensorRT → CUDA → CoreML → CPU
- Why: the same code runs on the Mac, the GPU server and in CI; the production runtime doesn't need torch
- Consequences: the post-processing has to stay in sync with RF-DETR's output format. It has only been checked against the library source → phase-4-b adds a parity test

## D-006 — FP16 by default; INT8 only if FP16 doesn't pass
- Date: 2026-10-20 · Phase: setup · Status: Accepted
- Decided by: agent
- Why: FP16 costs essentially no accuracy. On Orin, INT8 is only 10–17% faster than FP16 (slower on Thor), and drops 1.5–5 mAP.
- Revisit if: one device has to handle several cameras and FP16 can't keep up (phase-6-d)

## D-007 — ByteTrack + supervision LineZone for counting
- Date: 2026-10-20 · Phase: setup · Status: Accepted
- Decided by: agent
- Options: ByteTrack, BoT-SORT, OC-SORT (all from the `trackers` package)
- Decision: ByteTrack (`trackers.ByteTrackTracker`) to start
- Why: lightweight, no ReID model, so it fits the latency budget; it's the recommended default. BoT-SORT has higher HOTA on MOT17 (63.7) but costs more.
- Consequences: LineZone only counts detections that have a tracker_id; set `minimum_crossing_threshold` to filter jitter
- Revisit if: count accuracy in phase-6-e misses the KPI because of ID switches → try BoT-SORT

## D-008 — Tool stack: uv, ruff, DVC, MLflow, FiftyOne, pre-commit
- Date: 2026-10-20 · Phase: setup · Status: Accepted
- Decided by: user (uv + ruff + notebook) + agent (the rest)
- Decision: uv (env/lock), ruff (lint/format including notebooks), DVC (data/pipeline), MLflow 3.x (tracking/registry),
  FiftyOne (data QA), nbstripout + pre-commit, GitHub Actions
- Why: all open source; Ultralytics and RF-DETR integrate with MLflow directly; DVC stays open source after lakeFS acquired it
- Consequences: notebooks are for exploration only; reusable code goes in `src/`

## D-009 — Split evidently and fiftyone into separate environments
- Date: 2026-10-20 · Phase: setup · Status: Accepted
- Decided by: agent
- Context: evidently 0.7.x pins plotly<6, while fiftyone 1.22 needs plotly>=6, so they can't be installed together
- Decision: declare `conflicts = [[{extra="monitor"}, {group="data"}]]` in uv
- Consequences: FiftyOne is used on the dev machine; monitoring runs in the serving/monitoring service
- Revisit if: evidently supports plotly 6

## D-010 — Public datasets for learning; production uses COCO/CC BY + in-house Thai data
- Date: 2026-10-20 · Phase: setup · Status: Accepted
- Decided by: agent
- Context: MIO-TCD, KITTI, Cityscapes, Waymo and DAWN are non-commercial; BDD100K is for members only;
  for Waymo, a model trained on the data is also restricted
- Decision: workshop/learning may use every dataset. A model for business use is trained only on
  COCO / Open Images (CC BY) + in-house anonymised Thai footage
- Consequences: two sets of weights must be kept apart (learning vs commercial); record the license of every source in `docs/licenses.md`

## D-011 — Privacy by design: send only counts/boxes out of the device
- Date: 2026-10-20 · Phase: setup · Status: Accepted
- Decided by: agent
- Context: Thai PDPA; the PDPC's draft CCTV guideline (hearing 1–2 Apr 2026) flags LPR as a risk area. Fines in 2025 reached THB 7M.
- Decision: the service returns only boxes/classes/counts. No raw frames, faces or plates go into git, logs or MLflow. ANPR is a separate add-on that needs a DPIA.
- Consequences: frames collected for retraining (phase-8-c) must pass a blur step and a retention policy first

## D-012 — Harness: PLAN.md (Verify) + PROGRESS.md + DECISIONS.md instead of a JSON feature list
- Date: 2026-10-20 · Phase: setup · Status: Accepted
- Decided by: user (requested PLAN/PROGRESS/DECISIONS) + agent
- Context: Anthropic's long-running agent harness uses a JSON feature list with a `passes` field, because models are less likely to overwrite JSON than Markdown
- Decision: use PLAN.md with checkboxes + a Verify line for every sub-task, plus a CLAUDE.md rule that only the checkbox may be edited
- Why: the user reads and edits the plan directly; 44 sub-tasks are few enough to manage in Markdown
- Revisit if: agents keep editing goals/Verify criteria or ticking tasks without running Verify → move to `tasks.json`

## D-013 — Train on RTX 5070 12 GB with PyTorch cu130, via WSL2
- Date: 2026-10-20 · Phase: phase-1-f · Status: Accepted
- Decided by: user (RTX 5070) + agent (how to set it up)
- Context: the RTX 5070 is Blackwell (sm_120). PyTorch 2.14 dropped the cu128/cu129 wheels; CUDA 13.x builds cover Turing → Blackwell
- Options: A) native Windows B) WSL2 + Ubuntu 24.04 C) dual-boot Linux
- Decision: B. In `pyproject.toml`, the torch/torchvision index is `cu130` on Linux (WSL2 counts as linux). NVIDIA driver ≥ 580 is installed on Windows only.
- Why: `make`, `init.sh`, DVC and TensorRT behave the same as on Linux, there's no need to maintain a separate Windows path, and dual-booting isn't required
- Consequences: 12 GB VRAM → Small uses batch 4 × accum 4, Medium uses batch 2 × accum 8 + gradient checkpointing (effective batch 16 in both). Train with bf16. Keep the repo/data under `~/` in WSL, not `/mnt/c` (I/O is much slower there).
- Revisit if: PyTorch moves its default to CUDA 13.x beyond cu130, or the machine is changed to native Linux

## D-014 — Target device = RTX 5070 PC (edge server)
- Date: 2026-10-20 · Phase: phase-2-b · Status: Accepted
- Decided by: user
- Options: A) RTX 5070 PC B) Jetson Orin Nano Super C) Jetson Orin NX/AGX D) Cloud GPU
- Decision: A. The 5070 is used both for training and as the device that runs and is measured against the 60 FPS gate (`make bench`)
- Why: the easiest path to ≥ 60 FPS end-to-end; no cross-device TensorRT engine build; suits the workshop and a PoC. Small models take only a few ms per frame on a desktop GPU, leaving headroom for decode + tracking.
- Consequences: build the TensorRT engine on the 5070 itself (in WSL2). The latency budget in phase-2-b is set against the 5070. Benchmarks measured while training is running on the same machine don't count.
  If the deployment later moves to a Jetson, the FPS gate has to be re-measured on that device (a new decision record).
- Revisit if: one camera per site is needed (cost/power favours Jetson), or one machine must handle many cameras

## D-015 — DVC remote = local disk / NAS (on-premise)
- Date: 2026-10-20 · Phase: phase-1-d · Status: Accepted
- Decided by: user
- Options: A) local disk / NAS B) Google Drive C) S3 / R2 D) GCS
- Decision: A. Default path `~/dvc-store` on the 5070 machine (`make dvc-init DVC_REMOTE=...` changes it to a mounted NAS path)
- Why: free, fast, and keeps CCTV data on-premise, which matches D-011 (PDPA)
- Consequences: no off-site backup → back up the NAS/disk separately. GitHub-hosted CI can't `dvc pull` → the GPU bench job runs on a self-hosted runner on the same machine/LAN. Never commit a local path containing a username in `.dvc/config` → use `.dvc/config.local` for per-machine paths.
- Revisit if: there's a team outside the LAN or a cloud CI → move to S3/R2

## D-016 — No Thai footage yet: use public data + freely licensed clips first
- Date: 2026-10-20 · Phase: phase-3-b, phase-3-f · Status: Accepted
- Decided by: user
- Decision: phase-3 uses public datasets (D-010) + traffic clips whose license allows use (e.g. Pexels/Pixabay, CC BY videos) for evaluation, the counting demo and the bench.
  Plan real Thai footage collection in phase-3-f after phase-2-c (PDPA) is done.
- Why: the pipeline (phase-4) and the 60 FPS gate (phase-6) can start without Thai footage. Thai-specific classes (songthaew, tuk_tuk, pickup) still have little data.
- Consequences: test-set metrics don't yet reflect real Thai conditions → treat them as provisional. Don't use clips from YouTube or other copyrighted sources without permission.
  Record every clip's license in `docs/licenses.md`.
- Revisit if: real footage is obtained (from your own camera, a client or an agency) → add a record about the source and the data agreement

## D-017 — Checkpoint & run lineage: best by val mAP, resume, seed, test, MLflow artifacts + registry
- Date: 2026-10-20 · Phase: phase-4-f, phase-5-a, phase-5-f · Status: Accepted
- Decided by: user (asked for the standard practice) + agent (design)
- Context: RF-DETR already saves `checkpoint_best_total.pth` (best of EMA/regular), `last.ckpt` and interval archives, but the template didn't expose resume or a seed, didn't measure the test set, and didn't link weights to the run/code/data
- Decision:
  - Pick best by `best_model_metric: map` (val mAP50-95) — the same metric early stopping uses
  - `--resume auto|<path>` restarts from `last.ckpt` (power cut / OOM / WSL restart)
  - `seed: 42`; `run_test: true` measures the best checkpoint on a test split that was never used to choose the model
  - After training: load-test the best checkpoint → prune interval archives down to `keep_last_checkpoints` (never touches best/last)
    → log `model/` (checkpoint, train/data config, `model_meta.json` with sha256) to the **same** MLflow run, with tags `git_sha`, `git_dirty`, `dvc_data_rev`
  - The MLflow backend is `sqlite:///mlflow.db` (needed by the Model Registry, and the default in MLflow 3.x)
  - `vd-register` registers `model/` as a version and sets alias `candidate` → `production` only after eval + bench pass (phase-7-d)
- Why: you can trace back exactly which code + data produced each set of weights; a long training run on the 5070 doesn't have to start over; disk doesn't fill up
- Consequences: the best checkpoint is stored twice (DVC + MLflow artifact); turn off with `log_checkpoint_artifact: false` if space is tight, but then the registry can't be used.
  `vd-register` uses `MlflowClient.create_model_version` (not `mlflow.register_model`, which requires an MLflow flavor model)
- Revisit if: you need to serve directly from MLflow (`mlflow models serve`) → write a pyfunc wrapper

## D-018 — Pull datasets from HF / Kaggle / Roboflow with `vd-download`, separate from training inclusion
- Date: 2026-10-20 · Phase: phase-3-b · Status: Accepted
- Decided by: user (asked about HF/Kaggle) + agent
- Decision: `configs/data/sources.yaml` lists the sources (requires `license` + `commercial_ok`). `vd-download` downloads to a cache in `data/_downloads/` (not tracked by DVC),
  converts yolo / coco / hf_metadata → `data/raw/<source>/` (COCO with the original class names) and writes `SOURCE.json`. A source is used in training only after it is added to `source_mappings`
- Why: datasets on HF/Kaggle come in mixed formats and coarse classes (e.g. visaitech "4-wheeler"). Mapping straight to car would leave pickups/vans/buses unlabelled → the model learns them as background
- Consequences: the license an uploader tags on a mirror (e.g. ua_detrac on HF tagged CC BY 4.0) doesn't override the original license → mirrors whose original license is unverified are `commercial_ok: false`.
  Pin `revision` once you've chosen a dataset
- Revisit if: you need streaming of very large datasets → use the `datasets` library instead of `snapshot_download`
