"""Label / split QA (phase-3-e): duplicate images across splits and suspicious boxes.

    uv run vd-qa            # reads data/processed/coco, writes docs/label_issues.md

Two images are **duplicates** when their 64-bit difference hashes (dHash) differ in <= `MAX_DIST`
bits *and* their 64x48 grayscale thumbnails differ by <= `MAX_MAD` grey levels on average. dHash
alone also matches frames of one fixed camera that merely have a different car in them (same
background), so those are reported separately as **same scene** (a camera-level leak between
splits, not a duplicate). `drop_leaks` removes duplicates from valid/test (used by `vd-prepare`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

SPLITS = ("train", "valid", "test")
MAX_DIST = 4  # dHash bits (of 64)
MAX_MAD = 3.0  # mean absolute grey-level difference on a 64x48 thumbnail
SCENE_SAME = 0.4  # structure score at which two images show one camera view (see _scene_feature)
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def fingerprint(path: str | Path) -> tuple[int, np.ndarray]:
    """(64-bit dHash, 48x64 uint8 grayscale thumbnail) of an image."""
    with Image.open(path) as im:
        thumb = np.asarray(im.convert("L").resize((64, 48), Image.Resampling.LANCZOS))
    small = np.asarray(
        Image.fromarray(thumb).resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16
    )
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2), thumb


def dhash(path: str | Path) -> int:
    return fingerprint(path)[0]


def _hamming(hashes: np.ndarray, h: int) -> np.ndarray:
    diff = (hashes ^ np.uint64(h)).view(np.uint8).reshape(-1, 8)
    return _POPCOUNT[diff].sum(axis=1)


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def near_duplicates(hashes: list[int], max_dist: int = MAX_DIST) -> list[tuple[int, int, int]]:
    """All index pairs (i < j, distance) whose hashes differ in at most `max_dist` bits."""
    arr = np.array(hashes, dtype=np.uint64)
    pairs = []
    for i in range(len(arr) - 1):
        dist = _hamming(arr[i + 1 :], int(arr[i]))
        for k in np.nonzero(dist <= max_dist)[0]:
            pairs.append((i, i + 1 + int(k), int(dist[k])))
    return pairs


def drop_leaks(
    splits: dict[str, dict[str, Any]], max_dist: int = MAX_DIST, max_mad: float = MAX_MAD
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Remove from valid/test every image that duplicates one in an earlier split (train first).

    Reads `file_name` of each image as a path, so call it before the files are copied.
    """
    kept_hashes: list[int] = []
    kept_thumbs: list[np.ndarray] = []
    out: dict[str, dict[str, Any]] = {}
    removed: dict[str, int] = {}
    for name in SPLITS:
        ds = splits[name]
        arr = np.array(kept_hashes, dtype=np.uint64)
        drop: set[int] = set()
        fresh: list[tuple[int, np.ndarray]] = []
        for img in ds["images"]:
            h, thumb = fingerprint(img["file_name"])
            if kept_hashes:
                idx = np.nonzero(_hamming(arr, h) <= max_dist)[0]
                if any(mad(kept_thumbs[k], thumb) <= max_mad for k in idx):
                    drop.add(img["id"])
                    continue
            fresh.append((h, thumb))
        kept_hashes += [h for h, _ in fresh]
        kept_thumbs += [t for _, t in fresh]
        removed[name] = len(drop)
        out[name] = {
            **ds,
            "images": [i for i in ds["images"] if i["id"] not in drop],
            "annotations": [a for a in ds["annotations"] if a["image_id"] not in drop],
        }
    return out, removed


def assign_scene_ids(coco: dict[str, Any], sources: list[str], same: float = SCENE_SAME) -> int:
    """Give images of `sources` that have no `camera_id` a pseudo one per camera view.

    For stills from fixed cameras with no camera id (MIO-TCD): two images are the same view when
    their edge structure agrees (`_scene_feature`). Unlike a whole-image dHash or raw pixel
    comparison it survives a different vehicle in the frame, day vs night and moving shadows,
    which is how one camera ended up on both sides of the split (phase-3-e). Views are joined
    single-link and used as `camera_id`, so `group_split` keeps one camera inside one split.
    Reads `file_name` as a path (call before files are copied). Returns the number of views.
    """
    todo = [i for i in coco["images"] if i.get("source") in sources and "camera_id" not in i]
    if not todo:
        return 0
    feats = np.stack([_scene_feature(img["file_name"]) for img in todo])
    present = np.abs(feats)
    n_present = present.sum(axis=1)
    parent = list(range(len(todo)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for start in range(0, len(todo), 512):  # row blocks keep the score matrix small
        rows = slice(start, start + 512)
        union = n_present[rows, None] + n_present[None, :] - present[rows] @ present.T
        score = (feats[rows] @ feats.T) / np.maximum(union, 1)
        for r, c in zip(*np.nonzero(score >= same), strict=True):
            if start + r < c:
                parent[find(start + r)] = find(c)
    scene_of: dict[int, int] = {}
    for n, img in enumerate(todo):
        img["camera_id"] = f"scene{scene_of.setdefault(find(n), len(scene_of)):05d}"
    return len(scene_of)


def _scene_feature(path: str | Path) -> np.ndarray:
    """Edge-structure vector: sign of (64x48 grayscale - its blur) where contrast exists, else 0.

    The score of two such vectors is agreeing minus disagreeing edge pixels over pixels where
    either has an edge: about 0 for unrelated views, 0.4 or more for one camera.
    """
    with Image.open(path) as im:
        small = im.convert("L").resize((64, 48), Image.Resampling.BILINEAR)
    high_pass = np.asarray(small, dtype=np.float32) - np.asarray(
        small.filter(ImageFilter.GaussianBlur(4)), dtype=np.float32
    )
    return (np.sign(high_pass) * (np.abs(high_pass) > 3)).ravel()


def box_issues(coco: dict[str, Any]) -> list[tuple[str, int, str, str]]:
    """Suspicious boxes as (file_name, annotation id, class, issue)."""
    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)

    out = []
    for img_id, anns in by_image.items():
        img = images[img_id]
        w, h = img["width"], img["height"]
        for a in anns:
            x, y, bw, bh = a["bbox"]
            cls = names[a["category_id"]]
            if bw * bh < 16:
                out.append((img["file_name"], a["id"], cls, "tiny (< 16 px area)"))
            if bw * bh > 0.9 * w * h:
                out.append((img["file_name"], a["id"], cls, "huge (> 90% of the image)"))
            if x < -1 or y < -1 or x + bw > w + 1 or y + bh > h + 1:
                out.append((img["file_name"], a["id"], cls, "outside the image"))
        for i, a in enumerate(anns):
            for b in anns[i + 1 :]:
                if a["category_id"] == b["category_id"] and _iou(a["bbox"], b["bbox"]) > 0.9:
                    out.append(
                        (
                            img["file_name"],
                            b["id"],
                            names[b["category_id"]],
                            "duplicate box (IoU > 0.9)",
                        )
                    )
    return out


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def run(root: Path, max_dist: int = MAX_DIST, max_mad: float = MAX_MAD) -> dict[str, Any]:
    files: list[tuple[str, str]] = []  # (split, file name)
    hashes: list[int] = []
    thumbs: list[np.ndarray] = []
    issues: list[tuple[str, int, str, str]] = []
    n_images = {}
    for split in SPLITS:
        coco = json.loads((root / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        n_images[split] = len(coco["images"])
        for img in coco["images"]:
            files.append((split, img["file_name"]))
            h, t = fingerprint(root / split / img["file_name"])
            hashes.append(h)
            thumbs.append(t)
        issues += [(f"{split}/{f}", *rest) for f, *rest in box_issues(coco)]
    pairs = near_duplicates(hashes, max_dist)
    dup, scene = [], []
    for i, j, d in pairs:
        if files[i][0] == files[j][0]:
            continue
        (dup if mad(thumbs[i], thumbs[j]) <= max_mad else scene).append((files[i], files[j], d))
    return {
        "n_images": n_images,
        "inside": len(pairs) - len(dup) - len(scene),
        "duplicates": dup,
        "same_scene": scene,
        "same_scene_images": {
            s: len({b[1] for a, b, _ in scene if b[0] == s}) for s in ("valid", "test")
        },
        "issues": issues,
    }


def render(res: dict[str, Any], limit: int = 200) -> str:
    dup, scene, issues = res["duplicates"], res["same_scene"], res["issues"]
    sc = res["same_scene_images"]
    lines = [
        "# Label issues (phase-3-e)",
        "",
        "Generated by `uv run vd-qa` from `data/processed/coco`; re-run after every",
        "`make prepare`.",
        f"Images: {res['n_images']}. Duplicate = dHash distance <= {MAX_DIST} of 64 bits **and**",
        f"mean thumbnail difference <= {MAX_MAD} grey levels. Same scene = dHash match but a",
        "different picture (one fixed camera, different vehicles).",
        "",
        "## Duplicates and same-scene frames across splits",
        "",
        f"- **Duplicate pairs across splits: {len(dup)}**",
        "  (`vd-prepare` removes them from valid/test)",
        f"- **Same-scene pairs across splits: {len(scene)}**: {sc['valid']} valid and {sc['test']}",
        "  test images share a camera view with some train image. MIO-TCD has no camera id, so",
        f"  `vd-prepare` clusters its images by edge structure (score >= {SCENE_SAME}) into a",
        "  pseudo `camera_id` before splitting; a non-zero count here is the residual leak",
        f"- Pairs inside one split (harmless): {res['inside']}",
        "",
    ]
    if dup:
        lines += [
            "| Split A | Image A | Split B | Image B | dHash distance |",
            "|---|---|---|---|---|",
        ]
        lines += [f"| {a[0]} | {a[1]} | {b[0]} | {b[1]} | {d} |" for a, b, d in dup[:limit]]
        if len(dup) > limit:
            lines.append(f"\n... and {len(dup) - limit} more.")
        lines.append("")
    counts = Counter(i[3] for i in issues)
    lines += ["## Suspicious boxes", "", "| Issue | Count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())] or ["| none | 0 |"]
    lines += [
        "",
        'Review by eye on 2026-10-29 (8 of the "huge" boxes, all COCO): the labels are right but',
        "the pictures are consumer close-ups (a bus seen from inside, a bird on a car door, a cat",
        "next to a bicycle wheel), far from a road camera. They are a domain problem, not a",
        'labelling error; no box was edited or removed. "Tiny" boxes were not reviewed.',
    ]
    lines.append("")
    if issues:
        lines += ["| Image | Annotation id | Class | Issue |", "|---|---|---|---|"]
        lines += [f"| {f} | {a} | {c} | {t} |" for f, a, c, t in issues[:limit]]
        if len(issues) > limit:
            lines.append(f"\n... and {len(issues) - limit} more (run `vd-qa` for the full count).")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Duplicate and box QA for data/processed/coco")
    parser.add_argument("--root", default="data/processed/coco")
    parser.add_argument("--out", default="docs/label_issues.md")
    args = parser.parse_args()
    res = run(Path(args.root))
    Path(args.out).write_text(render(res), encoding="utf-8")
    sc = res["same_scene_images"]
    print(f"duplicate pairs across splits: {len(res['duplicates'])}")
    print(f"same-scene pairs across splits: {len(res['same_scene'])}")
    print(f"  images sharing a scene with train: valid {sc['valid']}, test {sc['test']}")
    print(f"suspicious boxes: {len(res['issues'])}  -> {args.out}")
