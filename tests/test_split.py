import pytest

from vehicle_detect.data.split import group_split


def test_no_camera_leaks_across_splits():
    images = [{"id": i, "file_name": f"{i}.jpg", "camera_id": f"cam{i % 10}"} for i in range(200)]
    anns = [{"id": i, "image_id": i, "category_id": 0} for i in range(200)]
    coco = {"images": images, "annotations": anns, "categories": [{"id": 0, "name": "car"}]}

    splits = group_split(coco, {"train": 0.8, "valid": 0.1, "test": 0.1}, "camera_id", seed=1)

    cams = {name: {img["camera_id"] for img in ds["images"]} for name, ds in splits.items()}
    assert not cams["train"] & cams["valid"]
    assert not cams["train"] & cams["test"]
    assert not cams["valid"] & cams["test"]
    assert sum(len(ds["images"]) for ds in splits.values()) == 200
    assert sum(len(ds["annotations"]) for ds in splits.values()) == 200


def _coco(images):
    anns = [{"id": i["id"], "image_id": i["id"], "category_id": 0} for i in images]
    return {"images": images, "annotations": anns, "categories": [{"id": 0, "name": "car"}]}


RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def test_same_camera_id_in_two_sources_is_two_groups():
    images = [
        {
            "id": i,
            "file_name": f"{i}.jpg",
            "camera_id": f"V00{i % 5}",
            "source": "a" if i < 100 else "b",
        }
        for i in range(200)
    ]
    splits = group_split(_coco(images), {"train": 0.6, "valid": 0.2, "test": 0.2}, "camera_id", 1)
    groups = {n: {(im["source"], im["camera_id"]) for im in d["images"]} for n, d in splits.items()}
    assert not groups["train"] & groups["valid"]
    assert not groups["train"] & groups["test"]
    # 10 (source, camera) groups, not 5: one camera_id may appear in several splits
    assert sum(len(g) for g in groups.values()) == 10


def test_empty_split_is_an_error():
    images = [{"id": i, "file_name": f"{i}.jpg", "camera_id": f"V{i % 2}"} for i in range(20)]
    with pytest.raises(ValueError, match="empty"):
        group_split(_coco(images), RATIOS, "camera_id", seed=1)


def test_missing_group_key_warns(caplog):
    images = [{"id": i, "file_name": f"{i}.jpg"} for i in range(30)]
    with caplog.at_level("WARNING"):
        group_split(_coco(images), RATIOS, "camera_id", seed=1)
    assert "camera_id" in caplog.text
