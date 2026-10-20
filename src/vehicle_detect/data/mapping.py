"""Remap COCO-format annotations from a source dataset onto the project taxonomy."""

from __future__ import annotations

from collections import Counter
from typing import Any


def build_category_map(
    source_categories: list[dict[str, Any]],
    name_mapping: dict[str, str | None],
    target_classes: list[str],
) -> dict[int, int | None]:
    """Return {source_category_id: target_category_id or None (drop)}.

    Unknown source names are dropped too, so a new dataset never silently adds classes.
    """
    target_ids = {name: i for i, name in enumerate(target_classes)}
    result: dict[int, int | None] = {}
    for cat in source_categories:
        target_name = name_mapping.get(cat["name"])
        if target_name is not None and target_name not in target_ids:
            raise ValueError(f"Mapping points to unknown target class: {target_name!r}")
        result[cat["id"]] = target_ids[target_name] if target_name else None
    return result


def remap_coco(
    coco: dict[str, Any],
    name_mapping: dict[str, str | None],
    target_classes: list[str],
) -> dict[str, Any]:
    """Return a new COCO dict with remapped categories; dropped boxes are removed."""
    cat_map = build_category_map(coco["categories"], name_mapping, target_classes)
    annotations = []
    for ann in coco["annotations"]:
        new_id = cat_map.get(ann["category_id"])
        if new_id is None:
            continue
        annotations.append({**ann, "category_id": new_id})
    categories = [
        {"id": i, "name": name, "supercategory": "vehicle"} for i, name in enumerate(target_classes)
    ]
    return {**coco, "annotations": annotations, "categories": categories}


def class_histogram(coco: dict[str, Any]) -> dict[str, int]:
    """Box count per class name — check imbalance before training."""
    names = {c["id"]: c["name"] for c in coco["categories"]}
    counts = Counter(names[a["category_id"]] for a in coco["annotations"])
    return {name: counts.get(name, 0) for name in names.values()}
