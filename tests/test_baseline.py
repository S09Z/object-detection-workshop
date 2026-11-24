import numpy as np
import supervision as sv

from vehicle_detect.baseline import remap_predictions, remap_targets

CLASSES = ["car", "pickup", "van", "bus", "truck", "motorcycle", "bicycle", "tuk_tuk", "songthaew"]


def _det(class_ids):
    n = len(class_ids)
    xyxy = np.tile([0.0, 0.0, 10.0, 10.0], (n, 1))
    return sv.Detections(xyxy=xyxy, class_id=np.array(class_ids), confidence=np.ones(n))


def test_predictions_keep_known_coco_classes_and_rewrite_ids():
    # COCO ids: 1 person, 2 bicycle, 3 car, 4 motorcycle, 6 bus, 8 truck, 10 traffic light
    pred = remap_predictions(_det([1, 2, 3, 4, 6, 8, 10]), CLASSES)
    assert pred.class_id.tolist() == [6, 0, 5, 3, 4]  # bicycle, car, motorcycle, bus, truck


def test_strict_drops_pickup_and_van_merged_relabels_them_car():
    target = _det([0, 1, 2, 3, 7])  # car, pickup, van, bus, tuk_tuk
    assert remap_targets(target, CLASSES, "strict").class_id.tolist() == [0, 3]
    assert remap_targets(target, CLASSES, "merged").class_id.tolist() == [0, 0, 0, 3]
