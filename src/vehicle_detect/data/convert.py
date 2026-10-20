"""Convert common dataset layouts into the project's raw layout:

    data/raw/<source>/images/*.jpg
    data/raw/<source>/annotations.coco.json      (original class names; mapping happens in prepare)

Supported input formats:
  coco         a COCO json + image folder (Roboflow "COCO" export, many HF repos)
  yolo         images/<split>/*.jpg + labels/<split>/*.txt (class cx cy w h, normalised)
               — HF / Kaggle / Roboflow
  mio_tcd      MIO-TCD localization: gt_train.csv rows (image_id, label, x1, y1, x2, y2)
  hf_metadata  HF imagefolder: <split>/metadata.jsonl, each row
               {"file_name", "objects": {"bbox", "category"}}
"""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CocoBuilder:
    def __init__(self, class_names: list[str]) -> None:
        self.categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []

    def add_image(self, file_name: str, width: int, height: int, **extra: Any) -> int:
        image_id = len(self.images)
        self.images.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height, **extra}
        )
        return image_id

    def add_box(self, image_id: int, category_id: int, x: float, y: float, w: float, h: float):
        self.annotations.append(
            {
                "id": len(self.annotations),
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }


def _copy_image(src: Path, images_dir: Path, prefix: str) -> tuple[str, int, int]:
    name = f"{prefix}_{src.name}" if prefix else src.name
    shutil.copy2(src, images_dir / name)
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"unreadable image: {src}")
    h, w = img.shape[:2]
    return name, w, h


def _camera_id(file_name: str, pattern: str | None) -> dict[str, str]:
    """Derive a camera/video id from the file name (e.g. r"^(V\\d+)" -> "V001") for group splits."""
    if pattern and (m := re.search(pattern, file_name)):
        return {"camera_id": m.group(1)}
    return {}


def yolo_to_coco(
    root: Path,
    out: Path,
    class_names: list[str],
    splits: list[str],
    camera_id_regex: str | None = None,
) -> dict:
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    coco = CocoBuilder(class_names)
    for split in splits:
        for img_path in sorted((root / "images" / split).glob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            name, w, h = _copy_image(img_path, images_dir, split)
            extra = _camera_id(img_path.name, camera_id_regex)
            image_id = coco.add_image(name, w, h, orig_split=split, **extra)
            label = root / "labels" / split / f"{img_path.stem}.txt"
            if not label.exists():
                continue  # background image
            for line in label.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
                coco.add_box(image_id, cls, (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h)
    return coco.to_dict()


def hf_metadata_to_coco(
    root: Path,
    out: Path,
    class_names: list[str],
    splits: list[str],
    camera_id_regex: str | None = None,
) -> dict:
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    coco = CocoBuilder(class_names)
    for split in splits:
        meta = root / split / "metadata.jsonl"
        for line in meta.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            name, w, h = _copy_image(root / split / row["file_name"], images_dir, split)
            extra = _camera_id(row["file_name"], camera_id_regex)
            image_id = coco.add_image(name, w, h, orig_split=split, **extra)
            objects = row.get("objects", {})
            for bbox, cat in zip(objects.get("bbox", []), objects.get("category", []), strict=True):
                coco.add_box(image_id, cat, *bbox)  # HF convention: COCO xywh in pixels
    return coco.to_dict()


def coco_passthrough(root: Path, out: Path, annotations: str, images: str) -> dict:
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    coco = json.loads((root / annotations).read_text(encoding="utf-8"))
    for img in coco["images"]:
        shutil.copy2(root / images / img["file_name"], images_dir / Path(img["file_name"]).name)
        img["file_name"] = Path(img["file_name"]).name
    return coco


def mio_tcd_to_coco(
    root: Path,
    out: Path,
    annotations: str,
    images: str,
    max_images: int,
    prefer_classes: list[str],
    seed: int = 42,
) -> dict:
    """MIO-TCD localization -> COCO, keeping `max_images` images.

    Half of the budget goes to images that contain a `prefer_classes` label (rare classes such as
    pickup_truck / work_van), the rest is filled at random, so the subset is not all `car`.
    Category names stay as in the CSV; `source_mappings` renames them later.
    """
    rows: dict[str, list[tuple[str, float, float, float, float]]] = {}
    with (root / annotations).open(encoding="utf-8", newline="") as f:
        for image_id, label, x1, y1, x2, y2 in csv.reader(f):
            rows.setdefault(image_id, []).append((label, *map(float, (x1, y1, x2, y2))))

    ids = sorted(rows)
    rng = random.Random(seed)
    rng.shuffle(ids)
    preferred = [i for i in ids if any(r[0] in prefer_classes for r in rows[i])]
    chosen = preferred[: max_images // 2]
    taken = set(chosen)
    chosen += [i for i in ids if i not in taken][: max_images - len(chosen)]

    names = sorted({r[0] for i in chosen for r in rows[i]})
    coco = CocoBuilder(names)
    ids_by_name = {n: i for i, n in enumerate(names)}
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for image_id in sorted(chosen):
        name, w, h = _copy_image(root / images / f"{image_id}.jpg", images_dir, "")
        img_id = coco.add_image(name, w, h)
        for label, x1, y1, x2, y2 in rows[image_id]:
            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box (seen in the real CSV)
            coco.add_box(img_id, ids_by_name[label], x1, y1, x2 - x1, y2 - y1)
    return coco.to_dict()
