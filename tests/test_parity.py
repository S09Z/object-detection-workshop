"""phase-4-b: `inference.postprocess` + `OnnxDetector` must agree with `RFDETR.predict()`.

Needs a trained checkpoint and the processed test split, so it is marked `slow` and skips
itself when either is missing:  uv run pytest -m slow tests/test_parity.py
Set PARITY_CHECKPOINT to pick a checkpoint (default: the newest runs/*/checkpoint_best_total.pth).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from vehicle_detect.config import load_config
from vehicle_detect.inference import OnnxDetector
from vehicle_detect.models import build_rfdetr

pytestmark = pytest.mark.slow

THRESHOLD = 0.5
MIN_IOU = 0.99
N_IMAGES = 10


def _checkpoint() -> Path:
    if env := os.environ.get("PARITY_CHECKPOINT"):
        return Path(env)
    found = sorted(
        Path("runs").glob("*/checkpoint_best_total.pth"), key=lambda p: p.stat().st_mtime
    )
    if not found:
        pytest.skip("no runs/*/checkpoint_best_total.pth (train first, or set PARITY_CHECKPOINT)")
    return found[-1]


def _images() -> list[Path]:
    files = sorted(Path("data/processed/coco/test").glob("*.jpg"))
    if len(files) < N_IMAGES:
        pytest.skip("data/processed/coco/test missing (run make prepare)")
    return files[:: len(files) // N_IMAGES][:N_IMAGES]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def test_onnx_detector_matches_predict(tmp_path):
    classes = load_config("configs/data/thai_traffic.yaml")["classes"]
    model = build_rfdetr("small", pretrain_weights=str(_checkpoint()), num_classes=len(classes))
    onnx_path = Path(
        model.export(output_dir=str(tmp_path), format="onnx", fp16=False, batch_size=1)
    )
    detector = OnnxDetector(str(onnx_path), len(classes), threshold=THRESHOLD)

    checked = 0
    for path in _images():
        frame = cv2.imread(str(path))
        ref = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=THRESHOLD)
        out = detector(frame)
        assert len(out) == len(ref), f"{path.name}: {len(out)} onnx vs {len(ref)} predict boxes"
        for box, cls in zip(ref.xyxy, ref.class_id, strict=True):
            ious = [_iou(box, o) for o in out.xyxy]
            best = int(np.argmax(ious))
            assert ious[best] >= MIN_IOU, f"{path.name}: best IoU {ious[best]:.4f}"
            assert int(out.class_id[best]) == int(cls), f"{path.name}: class differs"
            checked += 1
    assert checked > 0, "no detections above threshold: parity not exercised"
