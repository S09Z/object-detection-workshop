"""phase-4-a: COCO-pretrained RF-DETR (no training) on our test split — the number to beat.

    uv run --no-sync python -m vehicle_detect.baseline [--size small] [--limit N]

The pretrained model only knows COCO classes, so only these map onto our taxonomy
(COCO category id -> our class). pickup / van / tuk_tuk / songthaew have no COCO class.
Two views of the ground truth are reported:
  strict : pickup and van boxes are dropped; a COCO "car" prediction on a pickup counts as a
           false positive (the harshest reading, upper bound on how much training must add)
  merged : pickup and van are relabelled car (what a generic detector would call them)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision

from vehicle_detect.config import load_config
from vehicle_detect.models import build_rfdetr

# COCO category id (1-based, sparse) -> our class name
COCO_TO_OURS = {2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
MERGE_INTO_CAR = ("pickup", "van")


def remap_predictions(det: sv.Detections, classes: list[str]) -> sv.Detections:
    """Keep detections of COCO classes we know and rewrite class_id to our taxonomy."""
    ours = {coco_id: classes.index(name) for coco_id, name in COCO_TO_OURS.items()}
    keep = np.isin(det.class_id, list(ours))
    det = det[keep]
    det.class_id = np.array([ours[int(c)] for c in det.class_id], dtype=int)
    return det


def remap_targets(target: sv.Detections, classes: list[str], view: str) -> sv.Detections:
    """`strict` drops classes COCO lacks; `merged` also folds pickup/van into car."""
    known = {classes.index(n) for n in COCO_TO_OURS.values()}
    class_id = target.class_id.copy()
    if view == "merged":
        for name in MERGE_INTO_CAR:
            class_id[class_id == classes.index(name)] = classes.index("car")
    target = target[np.isin(class_id, list(known))]
    target.class_id = class_id[np.isin(class_id, list(known))]
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO-pretrained RF-DETR baseline")
    parser.add_argument("--size", default="small")
    parser.add_argument("--test-dir", default="data/processed/coco/test")
    parser.add_argument("--limit", type=int, default=0, help="first N images only (debug)")
    parser.add_argument("--out", default="reports/baseline_pretrained.json")
    args = parser.parse_args()

    classes = load_config("configs/data/thai_traffic.yaml")["classes"]
    test_dir = Path(args.test_dir)
    dataset = sv.DetectionDataset.from_coco(str(test_dir), str(test_dir / "_annotations.coco.json"))
    model = build_rfdetr(args.size)  # default weights = COCO-pretrained, 90 classes

    views = {v: MeanAveragePrecision() for v in ("strict", "merged")}
    n = 0
    for image_path, _image, target in dataset:
        pred = model.predict(
            cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), threshold=0.01
        )
        pred = remap_predictions(pred, classes)
        for view, metric in views.items():
            metric.update(pred, remap_targets(target, classes, view))
        n += 1
        if args.limit and n >= args.limit:
            break

    report: dict = {"model": f"rfdetr-{args.size} COCO-pretrained", "images": n, "threshold": 0.01}
    for view, metric in views.items():
        result = metric.compute()
        report[view] = {
            "map50_95": float(result.map50_95),
            "map50": float(result.map50),
            "per_class_ap50_95": {
                classes[int(c)]: float(ap.mean())
                for c, ap in zip(result.matched_classes, result.ap_per_class, strict=True)
            },
        }
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
