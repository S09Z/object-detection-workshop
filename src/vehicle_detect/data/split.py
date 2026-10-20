"""Group-aware train/valid/test split for COCO datasets (no frame leakage across splits)."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)


def group_split(
    coco: dict[str, Any],
    ratios: dict[str, float],
    group_by: str = "camera_id",
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Split images by a group key stored on each image (e.g. camera_id or video_id).

    Group ids are scoped by the image's `source` (when set), so the same camera id in two
    datasets is two groups. Images without the key are their own group (per-frame split,
    which leaks if they come from video) and are reported with a warning.
    Raises ValueError if any split ends up empty (too few groups for the ratios).
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ungrouped = 0
    for img in coco["images"]:
        if group_by in img:
            key = f"{img.get('source', '')}:{img[group_by]}"
        else:
            key = f"__img_{img['id']}"
            ungrouped += 1
        groups[key].append(img)
    if ungrouped:
        log.warning("%d images have no %r; they are split per frame", ungrouped, group_by)

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

    empty = [name for name, images in splits.items() if not images]
    if empty:
        raise ValueError(
            f"split(s) {empty} would be empty: only {len(groups)} {group_by} groups for "
            f"ratios {ratios}; add more cameras/videos or change the ratios"
        )

    out = {}
    for name, images in splits.items():
        ids = {img["id"] for img in images}
        out[name] = {
            **coco,
            "images": images,
            "annotations": [a for i in ids for a in anns_by_image[i]],
        }
    return out
