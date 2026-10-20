"""DVC stage `bench`: END-TO-END real-time gate (decode -> detect -> track), not model-only latency.

Pass criteria (workshop default): sustained FPS >= 60 AND p99 frame time <= 16.7 ms
over >= 5 minutes of footage. Exits with code 1 on failure so CI can block the release.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from vehicle_detect.config import cli_config
from vehicle_detect.inference import OnnxDetector


@dataclass
class LatencyReport:
    frames: int
    fps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    stage_mean_ms: dict[str, float]
    target_fps: float
    p99_budget_ms: float
    passed: bool
    provider: str = ""


def summarize(
    frame_ms: list[float],
    stages: dict[str, list[float]],
    target_fps: float = 60.0,
    p99_budget_ms: float = 1000 / 60,
) -> LatencyReport:
    arr = np.asarray(frame_ms, dtype=np.float64)
    fps = 1000.0 / arr.mean() if arr.size else 0.0
    p50, p95, p99 = (float(np.percentile(arr, q)) for q in (50, 95, 99))
    return LatencyReport(
        frames=int(arr.size),
        fps=round(fps, 1),
        p50_ms=round(p50, 2),
        p95_ms=round(p95, 2),
        p99_ms=round(p99, 2),
        stage_mean_ms={k: round(float(np.mean(v)), 2) for k, v in stages.items() if v},
        target_fps=target_fps,
        p99_budget_ms=round(p99_budget_ms, 2),
        passed=bool(fps >= target_fps and p99 <= p99_budget_ms),
    )


def main() -> None:
    cfg = cli_config("End-to-end real-time benchmark", "configs/bench/realtime.yaml")
    detector = OnnxDetector(cfg["model_path"], cfg["num_classes"], cfg["threshold"])
    tracker = None
    if cfg.get("with_tracking", True):
        from trackers import ByteTrackTracker

        tracker = ByteTrackTracker()

    cap = cv2.VideoCapture(cfg["video"])
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {cfg['video']}")

    warmup, max_frames = cfg.get("warmup_frames", 50), cfg.get("max_frames", 0)
    frame_ms: list[float] = []
    stages: dict[str, list[float]] = {"decode": [], "detect": [], "track": []}
    i = 0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()
        detections = detector(frame)
        t2 = time.perf_counter()
        if tracker is not None:
            tracker.update(detections)
        t3 = time.perf_counter()

        i += 1
        if i <= warmup:  # skip warm-up: CUDA/TensorRT lazy init, caches
            continue
        frame_ms.append((t3 - t0) * 1000)
        stages["decode"].append((t1 - t0) * 1000)
        stages["detect"].append((t2 - t1) * 1000)
        stages["track"].append((t3 - t2) * 1000)
        if max_frames and len(frame_ms) >= max_frames:
            break
    cap.release()

    if not frame_ms:
        raise SystemExit("No frames measured — video shorter than warm-up?")

    report = summarize(frame_ms, stages, cfg.get("target_fps", 60), cfg.get("p99_budget_ms", 16.7))
    report.provider = detector.provider
    out = Path(cfg.get("report_path", "reports/bench.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
