# Privacy — DPIA-lite (Thai PDPA)

> Scope: the intersection vehicle-counting system in `docs/problem.md` (1 fixed camera, on-premise
> RTX 5070 PC). This is an engineering data-flow review (D-011), **not legal advice**. Items marked
> `PROPOSED` need the user's confirmation. If real footage is ever collected, have the data
> controller review this before go-live.

## What data exists

| Data | Personal data? | Where it lives |
|---|---|---|
| Live camera stream | Yes: may show faces, plates, people, vehicles | Camera → the PC, in memory only |
| Recorded footage used for bench/count/retraining | Yes (same as above) | `data/` on the PC, DVC remote on local disk / NAS (D-015); never git |
| Public training datasets | Possibly (faces, plates in the images) | `data/raw/`, DVC only; licence per `docs/licenses.md` |
| Detections: boxes, class, confidence | No (no identity) | Returned by the service; bench/count reports |
| Counts per class and direction | No | `reports/counts.json`, service response |
| Model weights, configs, metrics | No | `runs/`, `models/` (DVC), MLflow |

## Data flow

```
camera --(LAN)--> decode --> detect --> track --> count --> counts / boxes --> consumer
                     |                                         (only output that leaves the PC)
                     +-- frames stay in RAM; not written to disk by the service
```

1. Frames are decoded and processed in memory on the PC. `vehicle_detect.serve.app` never writes an
   uploaded image to disk and returns only boxes, classes and confidences.
2. **What leaves the device: counts and boxes only.** No frames, crops, faces or plates.
3. Raw footage used for development stays on the PC / its DVC remote (local disk or NAS inside the
   same premises). It is never sent to a cloud service or pushed to GitHub.

## Where images must not appear

| Place | Rule | How it is enforced |
|---|---|---|
| git | no images, video, weights, data | `.gitignore` (`data/`, `models/*`, `runs/`, `reports/*`), pre-commit `no-large-or-data-files` hook for video/weights extensions |
| MLflow | log metrics, params, configs and checkpoints only; no sample frames | `train.py` logs `model/` (checkpoint, configs, metadata) only |
| Logs / test fixtures | no real frames | convention (CLAUDE.md §5); tests use synthetic arrays |
| Notebooks | outputs stripped before commit | `nbstripout` (pre-commit + `.gitattributes`) |
| Demo videos | `reports/count_demo.mp4` contains annotated raw frames | git-ignored; show only on the PC, do not publish unless blurred |

## Retention (`PROPOSED`)

| Item | Retention |
|---|---|
| Live frames | none (RAM only) |
| Recorded clips for dev/bench | keep only what the sub-task needs; delete when the sub-task is closed or after 90 days |
| Frames queued for labelling (phase-8-c) | face and plate blur first; delete the originals after labelling |
| Counts | long-term (not personal data) |

## CCTV notice

If the project films a place itself (phase-3-f, a real intersection), the controller must put up a
CCTV notice stating the purpose (traffic counting) and the contact person. Not needed while only
public datasets and freely licensed clips are used (D-016).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Plates or faces captured in footage or datasets | no ANPR or face features; keep footage on-device; blur before any sharing or labelling |
| Someone later logs frames "for debugging" | rule above + review in PRs; the service has no frame-writing code |
| Unauthenticated `/detect` endpoint exposes the service on the network | open item from the project review: add auth and bind to localhost before exposing (phase-7-a) |
| Demo video shared externally | blur or use licensed clips only |
| ANPR requested later | separate DPIA before any work (CLAUDE.md §5) |

## Open questions for the user

- Confirm the retention periods marked `PROPOSED`.
- Who is the data controller if real footage is collected (phase-3-f)?
