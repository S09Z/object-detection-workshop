# CLAUDE.md — vehicle-detect

Real-time vehicle detection + classification (9 Thai traffic classes). Hard target: **≥ 60 FPS
end-to-end** (decode → detect → track) on the target device, p99 frame time ≤ 16.7 ms.
Machines: dev = MacBook (Apple Silicon, MPS); train + target device = **RTX 5070 12 GB PC under WSL2 Ubuntu,
CUDA 13 / torch cu130** (D-013, D-014). DVC remote = local disk / NAS (D-015).
This is also a learning project for the full ML lifecycle, so explain reasoning briefly when it
teaches something. Reply in the user's language (usually Thai); keep code, comments and commits in English.

---

## 1. Session protocol (harness)

Every session starts with no memory. State lives in files, not in your context.

**Start of every session**
1. `pwd` — must be the repo root (contains `pyproject.toml`).
2. Read `PROGRESS.md` §1 *Current state* (and the last log entry), `git log --oneline -15`,
   and the `DECISIONS.md` index. Don't reverse an `Accepted` decision without writing a new one.
3. Read `PLAN.md`, pick the **first unchecked sub-task** whose dependencies are done.
   If the user named a sub-task (e.g. `phase-3-c`), do that one.
4. Run `./init.sh` — env sync + lint + tests. If it fails, fix that **before** new work.

**While working**
- **One sub-task at a time.** Don't start the next one in the same change.
- Each sub-task has a **Verify** line in `PLAN.md`. That is the definition of done.

**End of every sub-task**
1. Run its Verify command and make sure it passes. Show the output.
2. Tick the box in `PLAN.md` (`- [ ]` → `- [x]`). Edit **only** the checkbox.
   Never delete, weaken or rewrite a sub-task's goal or Verify criterion to make it pass.
   If a criterion is wrong, stop and propose the change to the user.
3. Update `PROGRESS.md`: **overwrite** §1 *Current state* (sub-task, status, health check,
   best metrics, blockers, open questions) and **append** a §2 log entry. Never edit old log entries.
4. If you chose between options that affect later work, append a record to `DECISIONS.md`
   (next `D-00N`, add it to the index). Needs-approval items (§5) are recorded as `Proposed`.
5. Commit: `phase-X-y: <what changed>` (add `[D-00N]` if relevant) — one sub-task per commit,
   tree left clean (lint + tests green).

**If blocked:** set status `blocked` in `PROGRESS.md` §1, write the question under
*Open questions for the user*, commit, and stop. Don't guess around a blocker.

**Never claim done** without the Verify output. "Should work" is not done.

---

## 2. Coding behaviour (Karpathy guidelines)

**Think before coding.** State assumptions. If a request has several readings, list them and ask;
don't silently pick one. If a simpler approach exists, say so. If something is unclear, stop and ask.

**Simplicity first.** Minimum code that solves the sub-task. No speculative features, abstractions
for single use, unrequested config options, or error handling for impossible cases.
If a senior engineer would call it overcomplicated, rewrite it shorter.

**Surgical changes.** Touch only what the sub-task needs. Match existing style. Don't refactor,
reformat or "improve" adjacent code. Remove only the dead code *your* change created; mention other
dead code instead of deleting it. Every changed line must trace back to the sub-task.

**Goal-driven execution.** Turn the task into a check first (a test, a metric, a command),
then loop until it passes. For bugs: write the failing test, then fix.

---

## 3. ML practice (Karpathy, "A Recipe for Training Neural Networks")

Neural-net training fails silently: bugs show up as slightly-worse metrics, not crashes. So:

1. **Become one with the data** before touching models: look at many images, class balance,
   box sizes, night/rain, label mistakes (`notebooks/01_`, `02_`).
2. **End-to-end skeleton + dumb baselines first.** Get the whole pipeline (data → train → eval →
   export → bench → count) running on a tiny subset with a pretrained model before tuning anything.
   Fix the random seed. Verify the loss at init. Visualize what goes *into* the network.
3. **Overfit one small batch** to ~zero loss. If it can't, there's a bug.
4. **Then generalize:** more/cleaner data first, augmentation second, bigger model last.
5. **Tune** with few, deliberate experiments, each logged to MLflow with a clear run name.
6. **Squeeze** (bigger resolution, longer training, ensembling) only at the end — and never at
   the cost of the 60 FPS gate.

Change **one thing per experiment**. Compare against the previous best run, not against memory.

---

## 4. Project conventions

| Area | Rule |
|---|---|
| Env | `uv` only. Add deps with `uv add` (or `uv add --group dev`). Commit `uv.lock`. Never `pip install`. |
| Lint/format | `make lint` / `make fmt` (ruff, includes notebooks). Must be clean before commit. |
| Code location | Reusable code lives in `src/vehicle_detect/`. Notebooks only explore and import from `src/`. |
| Notebooks | Numbered by lifecycle step. Never write to `data/processed/` from a notebook. Outputs stripped. |
| Config | Hyperparameters and paths go in `configs/*.yaml`, not in code. |
| Data & weights | DVC only (`data/`, `models/`, `runs/`). Never commit them to git. |
| Experiments | Every training run logs to MLflow (`make train`). |
| Checkpoints | Ship `checkpoint_best_total.pth` (best val mAP). Resume with `--resume auto`. Never edit or delete `best_*`/`last.ckpt` by hand. A run counts only if its MLflow run has `git_sha` + `dvc_data_rev` tags and `git_dirty=false` (D-017). |
| Datasets | Add sources to `configs/data/sources.yaml` with `license` + `commercial_ok`, then `make download`. Train on a source only after it's in `source_mappings` (D-018). |
| Splits | By camera/video (`split.group_by`), never by frame. |
| FPS | Measure end-to-end with `make bench` **on the target device**, ≥ 5 min of video, after warm-up. Model-only latency is not the metric. |
| TensorRT | Build engines on the device that runs them. Never commit engines. |
| Tests | `make test` (CPU). Hardware tests use `@pytest.mark.gpu`. |

**Commands:** `make help` lists everything. Pipeline: `make prepare → train → export → eval → bench → count`,
or `make repro` (DVC, re-runs changed stages only).

---

## 5. Guardrails — ask the user first

- **Licenses:** before adding any dataset or model, record its license in `docs/licenses.md`.
  MIO-TCD, KITTI, Cityscapes, Waymo, DAWN are non-commercial; BDD100K is members-only;
  Ultralytics YOLO (incl. fine-tuned weights) is AGPL-3.0. Default model is RF-DETR Nano/Small/Medium (Apache-2.0).
- **Privacy (Thai PDPA):** no raw CCTV frames, faces or license plates leave the device or enter git,
  logs, MLflow artifacts or test fixtures. Services return counts/boxes only. ANPR needs a DPIA first.
- **Destructive actions:** deleting data, `dvc gc`, force-push, rewriting git history,
  changing class taxonomy order (breaks every checkpoint), or changing a Verify criterion in `PLAN.md`.
- **New heavy dependencies** or a framework switch (e.g. RF-DETR → YOLO).

---

## 6. Known caveats in the current code

- `inference.postprocess` assumes RF-DETR ONNX outputs `dets` (normalised cxcywh) and `labels`
  (logits, background slot after the real classes). Verified only against the library source,
  not a real export — `phase-4-b` adds a parity test against `model.predict()`.
- `uv.lock` must be generated on a machine that can reach the PyTorch CUDA index (`cu130`).
  Never switch the index to cu128/cu126: PyTorch 2.14 dropped cu128, and cu126 has no sm_120 (RTX 50xx) kernels.
- RTX 5070 training configs (`configs/train/*_rtx5070.yaml`) have not run on real hardware yet; record
  peak VRAM and s/epoch in `PROGRESS.md` the first time they do.
- Don't run `make bench` while a training job is using the same GPU — the result is invalid.
- `evidently` (extra `monitor`) and `fiftyone` (group `data`) conflict on plotly; they're declared as
  conflicting in `pyproject.toml` and must live in separate environments.
