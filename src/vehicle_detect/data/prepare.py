"""DVC stage `prepare`: raw COCO datasets -> data/processed/coco/{train,valid,test}.

Expected raw layout (one folder per source dataset, key must match `source_mappings`):

    data/raw/<source>/images/*.jpg
    data/raw/<source>/annotations.coco.json

Output layout is what RF-DETR expects:

    data/processed/coco/train/_annotations.coco.json + images

`--commercial-only` builds a second split from sources with `commercial_ok: true` in
sources.yaml (D-010) into data/processed/coco_commercial. A source missing from sources.yaml
counts as non-commercial.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from vehicle_detect.config import load_config
from vehicle_detect.data.mapping import class_histogram, remap_coco
from vehicle_detect.data.qa import drop_leaks
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


def is_readable(path: str | Path) -> bool:
    """True if the image decodes fully (a truncated download opens but fails on load)."""
    try:
        with Image.open(path) as im:
            im.load()
    except OSError:
        return False
    return True


def drop_unreadable(ds: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Remove images that do not decode, and their annotations."""
    bad = {img["id"] for img in ds["images"] if not is_readable(img["file_name"])}
    if not bad:
        return ds, []
    names = [img["file_name"] for img in ds["images"] if img["id"] in bad]
    return {
        **ds,
        "images": [i for i in ds["images"] if i["id"] not in bad],
        "annotations": [a for a in ds["annotations"] if a["image_id"] not in bad],
    }, names


def commercial_sources(sources_cfg: dict[str, Any]) -> set[str]:
    """Names of sources whose licence allows commercial use (`commercial_ok: true`)."""
    return {s["name"] for s in sources_cfg.get("sources", []) if s.get("commercial_ok") is True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--config", default="configs/data/thai_traffic.yaml")
    parser.add_argument(
        "--commercial-only",
        action="store_true",
        help="use only commercial_ok sources -> data/processed/coco_commercial",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw, classes = Path(cfg["paths"]["raw"]), cfg["classes"]
    suffix = "_commercial" if args.commercial_only else ""
    out = Path(cfg["paths"]["processed"]) / f"coco{suffix}"
    allowed = None
    if args.commercial_only:
        allowed = commercial_sources(load_config(cfg["paths"]["sources"]))

    remapped = []
    for source, mapping in cfg["source_mappings"].items():
        if allowed is not None and source not in allowed:
            print(f"[skip] {source}: not commercial_ok (learning only)")
            continue
        ann_path = raw / source / "annotations.coco.json"
        if not ann_path.exists():
            print(f"[skip] {source}: {ann_path} not found")
            continue
        coco = json.loads(ann_path.read_text(encoding="utf-8"))
        ds = remap_coco(coco, mapping, classes)
        for img in ds["images"]:
            img["file_name"] = str(raw / source / "images" / img["file_name"])
            img.setdefault("source", source)
        ds, bad = drop_unreadable(ds)
        for name in bad:
            print(f"[warn] {source}: skipping unreadable image {name}")
        remapped.append(ds)
        print(f"[ok] {source}: {len(ds['images'])} images, {class_histogram(ds)}")

    if not remapped:
        raise SystemExit("No raw datasets found — see docs in this module.")

    split_cfg = cfg["split"]
    ratios = {k: split_cfg[k] for k in ("train", "valid", "test")}
    splits = group_split(merge(remapped), ratios, split_cfg["group_by"], split_cfg["seed"])
    splits, removed = drop_leaks(splits)  # a duplicate of a train image must not sit in valid/test
    if removed["valid"] or removed["test"]:
        print(f"[dedup] dropped duplicates of train images: valid {removed['valid']}")
        print(f"[dedup] dropped duplicates of earlier splits: test {removed['test']}")

    shutil.rmtree(out, ignore_errors=True)  # no stale files from earlier runs
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
    Path(f"reports/data_stats{suffix}.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
