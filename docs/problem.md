# Problem definition — vehicle-detect

> Status: **draft, default use case** (phase-2-a). Items marked `ASSUMPTION` are PLAN.md defaults
> and need the user's confirmation before phase-2-b / phase-5 rely on them.

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
| Business | Count accuracy vs a manual count of the same 15-minute clip, per class and direction | ≥ 95% total (`ASSUMPTION`, confirm in this phase) |
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

1. Count accuracy ≥ 95% against the manual 15-minute count (total, and no class below 90%, `ASSUMPTION`).
2. `make bench` reports `passed: true` (≥ 60 FPS and p99 ≤ 16.7 ms) on ≥ 5 minutes of footage,
   with no training job running on the same GPU.
3. Test mAP50-95 beats the pretrained baseline from phase-4-a, and no slice is more than
   0.15 mAP below the overall figure (`ASSUMPTION`).

## Out of scope (for now)

- License-plate reading (ANPR needs a DPIA first, see CLAUDE.md §5).
- Multi-camera fusion and cloud upload of frames.
- Thai footage of our own (none yet, D-016); public data stands in until phase-3-f.

## Open questions for the user

- Confirm the use case (traffic count at an intersection) or choose another.
- Confirm the 95% count-accuracy target and the per-class floor.
- Number of cameras and resolution (also needed by phase-2-b).
