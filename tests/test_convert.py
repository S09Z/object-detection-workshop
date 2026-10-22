import json

import cv2
import numpy as np
import pytest

from vehicle_detect.data.convert import hf_metadata_to_coco, yolo_to_coco


def _img(path, w=200, h=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))


def test_yolo_to_coco_pixels_and_camera_id(tmp_path):
    root, out = tmp_path / "yolo", tmp_path / "out"
    _img(root / "images/train/V001_0001.jpg")
    _img(root / "images/val/V002_0001.jpg")  # background image, no label file
    (root / "labels/train").mkdir(parents=True)
    (root / "labels/train/V001_0001.txt").write_text("1 0.5 0.5 0.2 0.4\n")

    coco = yolo_to_coco(root, out, ["a", "b"], ["train", "val"], camera_id_regex=r"(V\d{3})")

    assert [c["name"] for c in coco["categories"]] == ["a", "b"]
    assert len(coco["images"]) == 2
    assert {i["camera_id"] for i in coco["images"]} == {"V001", "V002"}
    (ann,) = coco["annotations"]
    assert ann["category_id"] == 1
    assert ann["bbox"] == pytest.approx([80.0, 30.0, 40.0, 40.0])  # 200x100 image
    assert (out / "images" / "train_V001_0001.jpg").exists()


def test_hf_metadata_to_coco(tmp_path):
    root, out = tmp_path / "hf", tmp_path / "out"
    _img(root / "train/a.jpg")
    row = {"file_name": "a.jpg", "objects": {"bbox": [[1, 2, 30, 40]], "category": [0]}}
    (root / "train/metadata.jsonl").write_text(json.dumps(row) + "\n")

    coco = hf_metadata_to_coco(root, out, ["car"], ["train"])

    assert coco["annotations"][0]["bbox"] == [1, 2, 30, 40]
    assert coco["images"][0]["width"] == 200


def test_mio_tcd_subset_favours_preferred_classes(tmp_path):
    import cv2
    import numpy as np

    from vehicle_detect.data.convert import mio_tcd_to_coco

    (tmp_path / "train").mkdir()
    rows = []
    for i in range(20):
        cv2.imwrite(str(tmp_path / "train" / f"{i:08d}.jpg"), np.zeros((40, 60, 3), np.uint8))
        label = "work_van" if i < 4 else "car"
        rows.append(f"{i:08d},{label},10,5,30,25")
    rows.append("00000019,car,7,7,7,20")  # zero-width box must be dropped
    (tmp_path / "gt_train.csv").write_text("\n".join(rows) + "\n")

    coco = mio_tcd_to_coco(tmp_path, tmp_path / "out", "gt_train.csv", "train", 8, ["work_van"], 1)

    names = {c["id"]: c["name"] for c in coco["categories"]}
    assert len(coco["images"]) == 8
    vans = [a for a in coco["annotations"] if names[a["category_id"]] == "work_van"]
    assert len(vans) == 4  # all 4 van images fit in the preferred half of the budget
    assert coco["annotations"][0]["bbox"] == [10.0, 5.0, 20.0, 20.0]  # x1y1x2y2 -> xywh
    assert all(a["bbox"][2] > 0 for a in coco["annotations"])
