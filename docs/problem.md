# Problem definition — vehicle-detect

> Status: use case and count-accuracy targets **confirmed by the user** (2026-10-20). Items still marked
> `ASSUMPTION` are PLAN.md defaults awaiting confirmation.

## Use case

Count vehicles by type as they cross a line at an intersection, from a fixed camera, on-premise
(RTX 5070 PC, D-014). Output is counts and boxes only; no frames leave the device (D-011).

## User

| Who | Needs |
|---|---|
| Traffic planner / road operator (`ASSUMPTION`) | Hourly counts per vehicle class and direction, to size signal timing and road works, replacing manual counting |
| Operator of the box | A service that stays up unattended and reports its own health (FPS, p99) |

## KPI

| Level | KPI | Target |
|---|---|---|
| Business | Count accuracy vs a manual count of the same 15-minute clip, per class and direction | ≥ 95% total, ≥ 90% per class |
| Business | Manual counting effort replaced | 0 staff-hours per camera for routine counts |
| Operational | End-to-end FPS on the target device (decode → detect → track) | ≥ 60 |
| Operational | p99 frame time | ≤ 16.7 ms |

## ML metric

| Metric | Why it maps to the KPI | Source |
|---|---|---|
| mAP50-95 on the held-out test split (9 classes) | Overall detection quality; compare against the pretrained baseline (phase-4-a) | `vd-evaluate` → `reports/eval.json` |
| Per-class AP50-95 | A class with low AP shows up as a count error for that class (e.g. songthaew, tuk_tuk) | `reports/eval.json` |
| Per-condition mAP (day / night / rain) | Production failures hide in slices | `reports/eval.json` |
| Track ID switches / line-crossing count error | Counting depends on tracking, not only detection | `vd-count` vs manual count |

## Acceptance threshold

The model is accepted for the demo when **all** hold on the target device:

1. Count accuracy ≥ 95% against the manual 15-minute count (total), and no class below 90%.
2. `make bench` reports `passed: true` (≥ 60 FPS and p99 ≤ 16.7 ms) on ≥ 5 minutes of footage,
   with no training job running on the same GPU.
3. Test mAP50-95 beats the pretrained baseline from phase-4-a, and no slice is more than
   0.15 mAP below the overall figure (`ASSUMPTION`).

## Out of scope (for now)

- License-plate reading (ANPR needs a DPIA first, see CLAUDE.md §5).
- Multi-camera fusion and cloud upload of frames.
- Thai footage of our own (none yet, D-016); public data stands in until phase-3-f.

## Open questions for the user

- Confirm the 0.15 mAP slice gap in acceptance threshold 3.

## Target hardware and latency budget (phase-2-b)

Device: RTX 5070 12 GB PC under WSL2 Ubuntu (D-014). Input: **1 camera, 1920×1080, 30 FPS source**
(confirmed by the user, 2026-10-20). The bench decodes a recorded file as fast
as the pipeline allows, so the 60 FPS gate is a 2× headroom over the 30 FPS live stream, and leaves
room for a second camera later.

Pipeline: decode → preprocess → model → postprocess → track → count. Per-frame allocation at 60 FPS
(1000 / 60 = 16.7 ms). These are **targets to be measured**, not measurements.

| Stage | Budget (ms) | Reported by `vd-bench` as | Notes |
|---|---|---|---|
| Decode 1080p | 3.0 | `decode` | threaded or NVDEC if CPU decode is too slow (phase-6-c) |
| Preprocess (resize, normalise) | 1.5 | inside `detect` | move to GPU if it exceeds this |
| Model (TensorRT FP16) | 8.0 | inside `detect` | phase-6-a: model-only ≤ 50% of this, i.e. ≤ 4.0 ms |
| Postprocess (top-k, boxes) | 0.5 | inside `detect` | |
| Track (ByteTrack) | 1.5 | `track` | |
| Count + overhead | 1.0 | inside `track` | line-crossing, bookkeeping |
| **Total** | **15.5** | | **≤ 16.7 ms, 1.2 ms margin** |

In `bench.json` terms: decode 3.0 + detect 10.0 (1.5 + 8.0 + 0.5) + track 2.5 (1.5 + 1.0) = 15.5 ms.
The model row is the largest, so if `detect` misses its budget the first levers are a smaller model
(Nano), lower input resolution, or INT8 (phase-6-d).

Conditions for a valid measurement: no training job on the same GPU, warm-up skipped, at least
5 minutes of footage, and p99 (not the mean) within 16.7 ms.
