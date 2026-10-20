import json

import pytest

from vehicle_detect.data import download

SOURCES = """
raw_dir: data/raw
sources:
  - {name: demo, enabled: true, hub: huggingface, license: MIT, commercial_ok: true}
"""


def _setup(tmp_path, monkeypatch):
    (tmp_path / "sources.yaml").write_text(SOURCES)
    out = tmp_path / "data/raw/demo"
    out.mkdir(parents=True)
    (out / "annotations.coco.json").write_text("OLD")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(download, "fetch", lambda src, dest: dest)
    monkeypatch.setattr("sys.argv", ["vd-download", "--config", str(tmp_path / "sources.yaml")])
    return out


def test_failed_convert_keeps_existing_dataset(tmp_path, monkeypatch):
    out = _setup(tmp_path, monkeypatch)

    def boom(src, root, out_dir):
        raise ValueError("unreadable image")

    monkeypatch.setattr(download, "convert", boom)
    with pytest.raises(ValueError):
        download.main()
    assert (out / "annotations.coco.json").read_text() == "OLD"
    assert not list(out.parent.glob(".*.tmp"))


def test_successful_convert_replaces_dataset(tmp_path, monkeypatch):
    out = _setup(tmp_path, monkeypatch)
    coco = {"images": [], "annotations": [], "categories": []}
    monkeypatch.setattr(download, "convert", lambda src, root, out_dir: coco)
    download.main()
    assert json.loads((out / "annotations.coco.json").read_text()) == coco
    assert (out / "SOURCE.json").exists()
    assert not list(out.parent.glob(".*.tmp"))
