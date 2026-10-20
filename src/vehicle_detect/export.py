"""DVC stage `export`: checkpoint -> ONNX (portable) or TensorRT/CoreML/OpenVINO (device-specific).

TensorRT engines are tied to the GPU + TensorRT version they were built with:
always build them ON the target device (Jetson, server), never on your laptop.
"""

from __future__ import annotations

import json

from vehicle_detect.config import cli_config, load_config
from vehicle_detect.models import build_rfdetr


def main() -> None:
    cfg = cli_config("Export model", "configs/export/trt_fp16.yaml")
    num_classes = len(load_config("configs/data/thai_traffic.yaml")["classes"])

    model = build_rfdetr(cfg["model"], pretrain_weights=cfg["checkpoint"], num_classes=num_classes)
    path = model.export(
        output_dir=cfg["output_dir"],
        format=cfg.get("format", "onnx"),
        fp16=cfg.get("fp16", True),
        batch_size=cfg.get("batch_size", 1),
    )
    print(json.dumps({"exported": str(path), "format": cfg.get("format", "onnx")}))


if __name__ == "__main__":
    main()
