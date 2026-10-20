"""Minimal FastAPI service (intro level). For multi-camera production use Triton / DeepStream.

Run:  uv run --extra serve uvicorn vehicle_detect.serve.app:app --port 8000
Privacy by design: returns boxes/classes only — never stores the uploaded frame.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from vehicle_detect.config import load_config
from vehicle_detect.inference import OnnxDetector

CLASSES = load_config(os.getenv("VD_DATA_CONFIG", "configs/data/thai_traffic.yaml"))["classes"]
detector = OnnxDetector(
    os.getenv("VD_MODEL_PATH", "models/export/inference_model.onnx"),
    num_classes=len(CLASSES),
    threshold=float(os.getenv("VD_THRESHOLD", "0.4")),
)
app = FastAPI(title="vehicle-detect")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": detector.provider}


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:  # noqa: B008
    data = np.frombuffer(await file.read(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Not an image")
    det = detector(frame)
    return {
        "detections": [
            {
                "box_xyxy": box.round(1).tolist(),
                "class": CLASSES[cid],
                "confidence": round(float(c), 3),
            }
            for box, cid, c in zip(det.xyxy, det.class_id, det.confidence, strict=True)
        ]
    }
