"""Group-aware train/valid/test split for COCO datasets (no frame leakage across splits)."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def group_split(
    coco: dict[str, Any],
    ratios: dict[str, float],
    group_by: str = "camera_id",
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Split images by a group key stored on each image (e.g. camera_id or video_id).

    Images without the key are treated as their own group.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for img in coco["images"]:
        groups[str(img.get(group_by, f"__img_{img['id']}"))].append(img)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    total = sum(ratios.values())
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in ratios}
    n_images = len(coco["images"])
    names = list(ratios)
    current = 0
    for key in keys:
        # move to the next split once the current one reaches its share
        while (
            current < len(names) - 1
            and len(splits[names[current]]) >= ratios[names[current]] / total * n_images
        ):
            current += 1
        splits[names[current]].extend(groups[key])

    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    out = {}
    for name, images in splits.items():
        ids = {img["id"] for img in images}
        out[name] = {
            **coco,
            "images": images,
            "annotations": [a for i in ids for a in anns_by_image[i]],
        }
    return out
