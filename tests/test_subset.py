from vehicle_detect.data.subset import sample, write


def _coco(n):
    return {
        "images": [{"id": i, "file_name": f"{i}.jpg"} for i in range(n)],
        "annotations": [{"id": i, "image_id": i, "category_id": 0} for i in range(n)],
        "categories": [{"id": 0, "name": "car"}],
    }


def test_sample_keeps_only_the_picked_images_annotations_and_is_seeded():
    a, b = sample(_coco(20), 5, seed=1), sample(_coco(20), 5, seed=1)
    assert [i["id"] for i in a["images"]] == [i["id"] for i in b["images"]]
    assert len(a["images"]) == 5
    assert {x["image_id"] for x in a["annotations"]} == {i["id"] for i in a["images"]}
    assert a["categories"] == _coco(1)["categories"]


def test_sample_asks_for_more_than_exists_returns_all():
    assert len(sample(_coco(3), 10, seed=0)["images"]) == 3


def test_write_copies_images_and_annotation_file(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "0.jpg").write_bytes(b"x")
    write(src, dst, sample(_coco(1), 1, seed=0))
    assert (dst / "0.jpg").read_bytes() == b"x"
    assert (dst / "_annotations.coco.json").exists()
