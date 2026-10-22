"""Download datasets from Hugging Face / Kaggle / Roboflow into data/raw/<source>/ (COCO layout).

    uv run vd-download                         # all sources with enabled: true
    uv run vd-download --only visaitech_mixed  # one source

Every source in configs/data/sources.yaml MUST declare `license` and `commercial_ok`.
Downloading does not include a source in training — add it to `source_mappings` in
configs/data/thai_traffic.yaml after EDA (phase-3-d). Afterwards: `dvc add data/raw`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from vehicle_detect.config import load_config
from vehicle_detect.data.convert import (
    coco_passthrough,
    hf_metadata_to_coco,
    mio_tcd_to_coco,
    yolo_to_coco,
)


def fetch(src: dict[str, Any], dest: Path) -> Path:
    """Download raw files into `dest` and return the folder that holds the dataset root."""
    hub = src["hub"]
    if hub == "huggingface":
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=src["repo_id"],
            repo_type="dataset",
            revision=src.get("revision"),  # pin a commit for reproducibility
            allow_patterns=src.get("allow_patterns"),
            local_dir=dest,
            token=os.getenv("HF_TOKEN"),
        )
    elif hub == "fiftyone_zoo":
        # Downloads only the images that contain `classes` (not the full 18 GB COCO train set).
        os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(dest.resolve())  # keep it out of ~/fiftyone
        import fiftyone.zoo as foz

        foz.download_zoo_dataset(
            src["zoo_name"],
            split=src["split"],
            label_types=["detections"],
            classes=src["classes"],
            max_samples=src.get("max_samples"),
            seed=src.get("seed", 42),
            shuffle=True,
        )
    elif hub == "http_tar":
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / Path(src["url"]).name
        # curl resumes a partial download (-C -); the archive is multi-GB
        subprocess.run(
            ["curl", "-L", "--fail", "-C", "-", "-o", str(archive), src["url"]], check=True
        )
        if not (dest / ".extracted").exists():
            with tarfile.open(archive) as tar:
                tar.extractall(dest, filter="data")
            (dest / ".extracted").touch()
    elif hub == "kaggle":
        from kaggle.api.kaggle_api_extended import KaggleApi  # needs ~/.kaggle/kaggle.json

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(src["dataset"], path=str(dest), unzip=True)
    elif hub == "roboflow":
        from roboflow import Roboflow  # uv add --group data roboflow

        rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
        project = rf.workspace(src["workspace"]).project(src["project"])
        project.version(src["version"]).download("coco", location=str(dest))
    else:
        raise ValueError(f"unknown hub: {hub}")
    return dest / src.get("subdir", "")


def class_names_for(src: dict[str, Any], root: Path) -> list[str]:
    if "class_names" in src:
        return src["class_names"]
    if (data_yaml := root / "data.yaml").exists():  # YOLO convention
        names = load_config(data_yaml)["names"]
        return list(names.values()) if isinstance(names, dict) else list(names)
    raise ValueError(f"{src['name']}: set class_names or provide data.yaml")


def convert(src: dict[str, Any], root: Path, out: Path) -> dict[str, Any]:
    fmt = src["format"]
    if fmt == "coco":
        return coco_passthrough(root, out, src["annotations"], src.get("images", "."))
    if fmt == "mio_tcd":
        return mio_tcd_to_coco(
            root,
            out,
            src["annotations"],
            src["images"],
            src["max_images"],
            src.get("prefer_classes", []),
            src.get("seed", 42),
        )
    names = class_names_for(src, root)
    splits = src.get("splits", ["train", "val"])
    regex = src.get("camera_id_regex")
    if fmt == "yolo":
        return yolo_to_coco(root, out, names, splits, regex)
    if fmt == "hf_metadata":
        return hf_metadata_to_coco(root, out, names, splits, regex)
    raise ValueError(f"unknown format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets into data/raw")
    parser.add_argument("--config", default="configs/data/sources.yaml")
    parser.add_argument("--only", default=None, help="source name")
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw = Path(cfg.get("raw_dir", "data/raw"))

    for src in cfg["sources"]:
        name = src["name"]
        if args.only and name != args.only:
            continue
        if not args.only and not src.get("enabled", False):
            continue
        if "license" not in src or "commercial_ok" not in src:
            raise SystemExit(f"{name}: license and commercial_ok are required (see D-010)")

        out = raw / name
        cache = Path("data/_downloads") / name  # outside data/raw: not DVC-tracked
        print(f"[{name}] {src['hub']} -> {out}  (license: {src['license']})")
        root = fetch(src, cache)
        # Convert beside the old dataset and swap at the end: a failure keeps the old data.
        tmp = out.parent / f".{name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            coco = convert(src, root, tmp)
            (tmp / "annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
            (tmp / "SOURCE.json").write_text(
                json.dumps({k: v for k, v in src.items() if k != "enabled"}, indent=2),
                encoding="utf-8",
            )
            shutil.rmtree(out, ignore_errors=True)
            tmp.rename(out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(
            f"[{name}] {len(coco['images'])} images, {len(coco['annotations'])} boxes, "
            f"classes: {[c['name'] for c in coco['categories']]}"
        )


if __name__ == "__main__":
    main()
