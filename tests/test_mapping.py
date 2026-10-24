import pytest

from vehicle_detect.config import load_config
from vehicle_detect.data.mapping import class_histogram, remap_coco

CLASSES = load_config("configs/data/thai_traffic.yaml")["classes"]


def _coco():
    return {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "categories": [
            {"id": 3, "name": "car"},
            {"id": 8, "name": "truck"},
            {"id": 1, "name": "person"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 1, "category_id": 8, "bbox": [0, 0, 10, 10]},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
        ],
    }


def test_remap_drops_unmapped_and_renumbers():
    out = remap_coco(_coco(), {"car": "car", "truck": "truck"}, CLASSES)
    assert [a["category_id"] for a in out["annotations"]] == [
        CLASSES.index("car"),
        CLASSES.index("truck"),
    ]
    assert len(out["categories"]) == len(CLASSES)
    assert class_histogram(out)["car"] == 1


def test_unknown_target_class_raises():
    with pytest.raises(ValueError, match="unknown target class"):
        remap_coco(_coco(), {"car": "spaceship"}, CLASSES)


def test_every_config_mapping_points_to_a_real_class():
    cfg = load_config("configs/data/thai_traffic.yaml")
    for source, mapping in cfg["source_mappings"].items():
        for target in mapping.values():
            assert target is None or target in cfg["classes"], (source, target)


def test_remap_drops_crowd_boxes():
    coco = _coco()
    coco["annotations"][0]["iscrowd"] = 1  # one box covering a group of cars
    out = remap_coco(coco, {"car": "car", "truck": "truck"}, CLASSES)
    assert [a["category_id"] for a in out["annotations"]] == [CLASSES.index("truck")]
