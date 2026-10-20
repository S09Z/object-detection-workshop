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
