"""Framework-free ONNX Runtime detector used by benchmark, counting and serving.

Keeping inference outside PyTorch means the exact same code runs on a Mac (CoreML EP),
a GPU server (TensorRT / CUDA EP) and in CI (CPU EP).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PREFERRED_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]


def preprocess(frame_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """BGR uint8 HxWx3 -> float32 1x3xHxW, RGB, ImageNet-normalised (RF-DETR contract)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    x = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def postprocess(
    boxes_cxcywh: np.ndarray,
    logits: np.ndarray,
    image_hw: tuple[int, int],
    num_classes: int,
    threshold: float = 0.4,
    top_k: int = 300,
) -> sv.Detections:
    """Decode RF-DETR outputs (queries Q): boxes (Q,4) normalised cxcywh, logits (Q,C+extra).

    RF-DETR is NMS-free: take sigmoid scores, top-k over (query, class) pairs, threshold.
    Only the first `num_classes` logit columns are used (drops the background slot).
    """
    scores = 1.0 / (1.0 + np.exp(-logits[:, :num_classes]))
    flat = scores.reshape(-1)
    k = min(top_k, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[flat[idx] >= threshold]
    query, class_id = np.divmod(idx, num_classes)

    h, w = image_hw
    cx, cy, bw, bh = boxes_cxcywh[query].T
    xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
    return sv.Detections(
        xyxy=xyxy.astype(np.float32).reshape(-1, 4),
        confidence=flat[idx].astype(np.float32),
        class_id=class_id.astype(int),
    )


@dataclass
class OnnxDetector:
    model_path: str
    num_classes: int
    threshold: float = 0.4

    def __post_init__(self) -> None:
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [p for p in PREFERRED_PROVIDERS if p in available]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.height, self.width = inp.shape
        names = [o.name for o in self.session.get_outputs()]
        self.boxes_idx = next(i for i, n in enumerate(names) if "dets" in n)
        self.logits_idx = next(i for i, n in enumerate(names) if "labels" in n)
        self.provider = self.session.get_providers()[0]

    def __call__(self, frame_bgr: np.ndarray) -> sv.Detections:
        outputs = self.session.run(
            None, {self.input_name: preprocess(frame_bgr, self.height, self.width)}
        )
        return postprocess(
            outputs[self.boxes_idx][0],
            outputs[self.logits_idx][0],
            frame_bgr.shape[:2],
            self.num_classes,
            self.threshold,
        )
