import json
import sys

import numpy as np
import yaml
from PIL import Image

from vehicle_detect.config import load_config
from vehicle_detect.data import prepare


def test_commercial_sources_only_explicit_true():
    cfg = {
        "sources": [
            {"name": "a", "commercial_ok": True},
            {"name": "b", "commercial_ok": False},
            {"name": "c"},
            {"name": "d", "commercial_ok": "yes"},
        ]
    }
    assert prepare.commercial_sources(cfg) == {"a"}


def test_every_mapped_source_is_in_sources_yaml():
    thai = load_config("configs/data/thai_traffic.yaml")
    known = {s["name"] for s in load_config(thai["paths"]["sources"])["sources"]}
    assert set(thai["source_mappings"]) <= known


def _write_source(raw, name, n_cams):
    (raw / name / "images").mkdir(parents=True)
    images, anns = [], []
    for i in range(n_cams):
        # distinct pictures: identical ones would be removed as duplicates across splits
        rng = np.random.default_rng(sum(map(ord, name)) * 100 + i)
        noise = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(noise).save(raw / name / "images" / f"{i}.jpg")
        images.append({"id": i, "file_name": f"{i}.jpg", "camera_id": f"{name}{i}"})
        anns.append({"id": i, "image_id": i, "category_id": 1, "bbox": [0, 0, 5, 5], "area": 25})
    coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "Car"}]}
    (raw / name / "annotations.coco.json").write_text(json.dumps(coco))


def test_commercial_only_split_excludes_learning_sources(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_source(raw, "free", 8)
    _write_source(raw, "nc", 8)
    sources = {"sources": [{"name": "free", "commercial_ok": True}, {"name": "nc"}]}
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(sources))
    cfg = {
        "classes": ["car"],
        "source_mappings": {"free": {"Car": "car"}, "nc": {"Car": "car"}},
        "paths": {
            "raw": str(raw),
            "processed": str(tmp_path / "processed"),
            "sources": str(tmp_path / "sources.yaml"),
        },
        "split": {"train": 0.5, "valid": 0.25, "test": 0.25, "seed": 0, "group_by": "camera_id"},
    }
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(sys, "argv", ["vd-prepare", "--config", "cfg.yaml", "--commercial-only"])
    prepare.main()
    stats = json.loads((tmp_path / "reports/data_stats_commercial.json").read_text())
    assert sum(s["images"] for s in stats.values()) == 8
    files = list((tmp_path / "processed/coco_commercial").rglob("*.jpg"))
    assert files and all(f.name.startswith("free_") for f in files)
    assert not (tmp_path / "processed/coco").exists()

    monkeypatch.setattr(sys, "argv", ["vd-prepare", "--config", "cfg.yaml"])
    prepare.main()
    stats = json.loads((tmp_path / "reports/data_stats.json").read_text())
    assert sum(s["images"] for s in stats.values()) == 16


def test_unreadable_image_is_skipped_and_stale_output_is_cleared(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_source(raw, "free", 8)
    # image 3 is truncated
    good = (raw / "free/images/3.jpg").read_bytes()
    (raw / "free/images/3.jpg").write_bytes(good[: len(good) // 2])
    stale = tmp_path / "processed/coco/train/old_run.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    cfg = {
        "classes": ["car"],
        "source_mappings": {"free": {"Car": "car"}},
        "paths": {"raw": str(raw), "processed": str(tmp_path / "processed")},
        "split": {"train": 0.5, "valid": 0.25, "test": 0.25, "seed": 0, "group_by": "camera_id"},
    }
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vd-prepare", "--config", "cfg.yaml"])
    prepare.main()

    stats = json.loads((tmp_path / "reports/data_stats.json").read_text())
    assert sum(s["images"] for s in stats.values()) == 7
    assert not stale.exists()
    assert not list((tmp_path / "processed").rglob("free_3.jpg"))
