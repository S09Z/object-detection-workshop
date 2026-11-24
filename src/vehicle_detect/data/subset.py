"""Small COCO subsets of data/processed/coco for sanity runs (overfit one batch, smoke pipeline).

    uv run python -m vehicle_detect.data.subset --out data/processed/overfit16 --train 16 --same
    uv run python -m vehicle_detect.data.subset --out data/processed/smoke \
        --train 500 --valid 100 --test 100

`--same` writes the train picks into valid and test too, so an overfit run is also evaluated on
exactly the images it trained on.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


def sample(coco: dict[str, Any], n: int, seed: int) -> dict[str, Any]:
    """n random images (seeded) with their annotations; categories are kept whole."""
    images = random.Random(seed).sample(coco["images"], min(n, len(coco["images"])))
    ids = {i["id"] for i in images}
    return {
        **coco,
        "images": images,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
    }


def write(src: Path, dst: Path, coco: dict[str, Any]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for img in coco["images"]:
        shutil.copy2(src / img["file_name"], dst / img["file_name"])
    (dst / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a small COCO subset")
    parser.add_argument("--src", default="data/processed/coco")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--valid", type=int, default=0)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--same", action="store_true", help="valid and test = the train picks")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    src, out = Path(args.src), Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    picks = {}
    for name, n in (("train", args.train), ("valid", args.valid), ("test", args.test)):
        coco = json.loads((src / name / "_annotations.coco.json").read_text(encoding="utf-8"))
        picks[name] = sample(coco, n, args.seed)
        write(src / name, out / name, picks[name])
    if args.same:
        for name in ("valid", "test"):
            shutil.rmtree(out / name)
            write(src / "train", out / name, picks["train"])
    for name in ("train", "valid", "test"):
        n_img = len(json.loads((out / name / "_annotations.coco.json").read_text())["images"])
        print(f"{name}: {n_img} images")


if __name__ == "__main__":
    main()
