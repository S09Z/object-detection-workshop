"""DVC stage `prepare`: raw COCO datasets -> data/processed/coco/{train,valid,test}.

Expected raw layout (one folder per source dataset, key must match `source_mappings`):

    data/raw/<source>/images/*.jpg
    data/raw/<source>/annotations.coco.json

Output layout is what RF-DETR expects:

    data/processed/coco/train/_annotations.coco.json + images
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from vehicle_detect.config import cli_config
from vehicle_detect.data.mapping import class_histogram, remap_coco
from vehicle_detect.data.split import group_split


def merge(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge already-remapped COCO dicts, re-numbering image and annotation ids."""
    images, annotations = [], []
    next_img, next_ann = 0, 0
    for ds in datasets:
        id_map = {}
        for img in ds["images"]:
            id_map[img["id"]] = next_img
            images.append({**img, "id": next_img})
            next_img += 1
        for ann in ds["annotations"]:
            annotations.append({**ann, "id": next_ann, "image_id": id_map[ann["image_id"]]})
            next_ann += 1
    return {"images": images, "annotations": annotations, "categories": datasets[0]["categories"]}


def main() -> None:
    cfg = cli_config("Prepare training data", "configs/data/thai_traffic.yaml")
    raw, out = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["processed"]) / "coco"
    classes = cfg["classes"]

    remapped = []
    for source, mapping in cfg["source_mappings"].items():
        ann_path = raw / source / "annotations.coco.json"
        if not ann_path.exists():
            print(f"[skip] {source}: {ann_path} not found")
            continue
        coco = json.loads(ann_path.read_text(encoding="utf-8"))
        ds = remap_coco(coco, mapping, classes)
        for img in ds["images"]:
            img["file_name"] = str(raw / source / "images" / img["file_name"])
            img.setdefault("source", source)
        remapped.append(ds)
        print(f"[ok] {source}: {len(ds['images'])} images, {class_histogram(ds)}")

    if not remapped:
        raise SystemExit("No raw datasets found — see docs in this module.")

    split_cfg = cfg["split"]
    ratios = {k: split_cfg[k] for k in ("train", "valid", "test")}
    splits = group_split(merge(remapped), ratios, split_cfg["group_by"], split_cfg["seed"])

    stats = {}
    for name, ds in splits.items():
        split_dir = out / name
        split_dir.mkdir(parents=True, exist_ok=True)
        for img in ds["images"]:
            src = Path(img["file_name"])
            dst_name = f"{img['source']}_{src.name}"
            shutil.copy2(src, split_dir / dst_name)
            img["file_name"] = dst_name
        (split_dir / "_annotations.coco.json").write_text(json.dumps(ds), encoding="utf-8")
        stats[name] = {"images": len(ds["images"]), "classes": class_histogram(ds)}

    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
