import json
import sys

import yaml

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
        (raw / name / "images" / f"{i}.jpg").write_bytes(b"x")
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
