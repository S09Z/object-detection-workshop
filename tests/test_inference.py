import numpy as np

from vehicle_detect.inference import postprocess, preprocess


def test_preprocess_shape_and_dtype():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    x = preprocess(frame, 512, 512)
    assert x.shape == (1, 3, 512, 512)
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]


def test_postprocess_decodes_one_confident_box():
    q, n_cls = 300, 9
    boxes = np.full((q, 4), 0.1, dtype=np.float32)
    logits = np.full((q, n_cls + 1), -10.0, dtype=np.float32)
    boxes[7] = [0.5, 0.5, 0.2, 0.4]  # centre of the image
    logits[7, 3] = 5.0  # class 3 (bus), sigmoid ~ 0.993
    logits[:, n_cls] = 10.0  # background slot must be ignored

    det = postprocess(boxes, logits, (720, 1280), num_classes=n_cls, threshold=0.5)

    assert len(det) == 1
    assert det.class_id[0] == 3
    np.testing.assert_allclose(det.xyxy[0], [512, 216, 768, 504], atol=1e-3)


def test_postprocess_empty_when_below_threshold():
    det = postprocess(np.zeros((10, 4)), np.full((10, 10), -10.0), (100, 100), 9, 0.5)
    assert len(det) == 0
