"""DVC stage `evaluate`: mAP + per-class AP + per-condition slices on the test split.

Slices: add an optional "condition" field to each COCO image (e.g. day/night/rain) and the
report will include mAP per condition — that is where production failures usually hide.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import supervision as sv
from supervision.metrics import MeanAveragePrecision

from vehicle_detect.config import cli_config
from vehicle_detect.inference import OnnxDetector


def main() -> None:
    cfg = cli_config("Evaluate exported model", "configs/bench/realtime.yaml")
    test_dir = Path(cfg.get("test_dir", "data/processed/coco/test"))
    ann_path = test_dir / "_annotations.coco.json"

    dataset = sv.DetectionDataset.from_coco(str(test_dir), str(ann_path))
    conditions = {
        img["file_name"]: img.get("condition", "all")
        for img in json.loads(ann_path.read_text(encoding="utf-8"))["images"]
    }
    detector = OnnxDetector(cfg["model_path"], cfg["num_classes"], threshold=0.001)

    overall = MeanAveragePrecision()
    by_condition: dict[str, MeanAveragePrecision] = defaultdict(MeanAveragePrecision)
    for image_path, _image, target in dataset:
        pred = detector(cv2.imread(image_path))
        overall.update(pred, target)
        by_condition[conditions.get(Path(image_path).name, "all")].update(pred, target)

    result = overall.compute()
    report = {
        "map50_95": float(result.map50_95),
        "map50": float(result.map50),
        "per_class_ap50_95": {
            dataset.classes[int(c)]: float(ap.mean())
            for c, ap in zip(result.matched_classes, result.ap_per_class, strict=True)
        },
        "per_condition_map50_95": {k: float(m.compute().map50_95) for k, m in by_condition.items()},
    }
    out = Path("reports/eval.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
