import json

import numpy as np
from PIL import Image

from vehicle_detect.data import qa


def _save(path, seed, size=(64, 48)):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


def test_dhash_matches_resized_copy_but_not_a_different_image(tmp_path):
    _save(tmp_path / "a.png", 1, (200, 150))
    Image.open(tmp_path / "a.png").resize((100, 75)).save(tmp_path / "a_small.png")
    _save(tmp_path / "b.png", 2, (200, 150))
    ha, hs, hb = (qa.dhash(tmp_path / n) for n in ("a.png", "a_small.png", "b.png"))
    assert bin(ha ^ hs).count("1") <= 8
    assert bin(ha ^ hb).count("1") > 16


def test_run_flags_leak_between_train_and_test(tmp_path):
    for name, seed in (("train", 1), ("test", 1), ("valid", 2)):
        d = tmp_path / name
        d.mkdir()
        _save(d / "x.png", seed)
    imgs = [{"id": 1, "file_name": "x.png", "width": 64, "height": 48}]
    anns = [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [5, 5, 20, 20]}]
    for name in ("train", "valid", "test"):
        coco = {"images": imgs, "annotations": anns, "categories": [{"id": 0, "name": "car"}]}
        (tmp_path / name / "_annotations.coco.json").write_text(json.dumps(coco))

    res = qa.run(tmp_path, max_dist=4)

    assert [(a[0], b[0]) for a, b, _ in res["duplicates"]] == [("train", "test")]


def test_drop_leaks_removes_only_real_duplicates_from_valid_and_test(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (48, 64), dtype=np.uint8)
    other = rng.integers(0, 255, (48, 64), dtype=np.uint8)
    for name, arr in (("tr", base), ("same", base), ("diff", other)):
        Image.fromarray(arr).save(tmp_path / f"{name}.png")

    def ds(names):
        images = [{"id": i, "file_name": str(tmp_path / f"{n}.png")} for i, n in enumerate(names)]
        anns = [{"id": i, "image_id": i} for i in range(len(names))]
        return {"images": images, "annotations": anns, "categories": []}

    splits = {"train": ds(["tr"]), "valid": ds(["same", "diff"]), "test": ds(["same"])}
    out, removed = qa.drop_leaks(splits)

    assert removed == {"train": 0, "valid": 1, "test": 1}
    assert [i["file_name"][-8:] for i in out["valid"]["images"]] == ["diff.png"]
    assert out["valid"]["annotations"] == [{"id": 1, "image_id": 1}]
    assert out["test"]["images"] == []


def test_box_issues_finds_tiny_huge_outside_and_duplicate_boxes():
    coco = {
        "images": [{"id": 1, "file_name": "i.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 0, "name": "car"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 2, 2]},
            {"id": 2, "image_id": 1, "category_id": 0, "bbox": [0, 0, 99, 99]},
            {"id": 3, "image_id": 1, "category_id": 0, "bbox": [90, 90, 30, 30]},
            {"id": 4, "image_id": 1, "category_id": 0, "bbox": [40, 40, 20, 20]},
            {"id": 5, "image_id": 1, "category_id": 0, "bbox": [40, 40, 20, 21]},
        ],
    }
    issues = {t for _, _, _, t in qa.box_issues(coco)}
    assert issues == {
        "tiny (< 16 px area)",
        "huge (> 90% of the image)",
        "outside the image",
        "duplicate box (IoU > 0.9)",
    }
