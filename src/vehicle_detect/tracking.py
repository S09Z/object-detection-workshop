"""Vehicle counting by class across a line: detect -> ByteTrack -> supervision LineZone.

LineZone only counts detections that carry a tracker_id, so tracking is mandatory here.
Output: annotated demo video + per-class in/out counts (compare against a manual count
to get the business KPI "count accuracy").
"""

from __future__ import annotations

import json
from pathlib import Path

import supervision as sv

from vehicle_detect.config import cli_config, load_config
from vehicle_detect.inference import OnnxDetector


def count_video(cfg: dict) -> dict:
    classes = load_config("configs/data/thai_traffic.yaml")["classes"]
    detector = OnnxDetector(cfg["model_path"], cfg["num_classes"], cfg["threshold"])

    from trackers import ByteTrackTracker

    video_info = sv.VideoInfo.from_video_path(cfg["video"])
    tracker = ByteTrackTracker(frame_rate=video_info.fps)
    line = sv.LineZone(
        start=sv.Point(*cfg["line_start"]),
        end=sv.Point(*cfg["line_end"]),
        minimum_crossing_threshold=cfg.get("minimum_crossing_threshold", 1),
    )
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    line_annotator = sv.LineZoneAnnotator()

    out_path = Path(cfg["output_video"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sv.VideoSink(str(out_path), video_info) as sink:
        for frame in sv.get_video_frames_generator(cfg["video"]):
            detections = tracker.update(detector(frame))
            line.trigger(detections)
            tracker_ids = detections.tracker_id if detections.tracker_id is not None else []
            labels = [
                f"#{tid} {classes[cid]}"
                for tid, cid in zip(tracker_ids, detections.class_id, strict=False)
            ]
            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels)
            sink.write_frame(line_annotator.annotate(annotated, line))

    def named(d: dict) -> dict[str, int]:
        return {classes[k]: v for k, v in d.items() if k is not None}

    return {
        "in_total": line.in_count,
        "out_total": line.out_count,
        "in_per_class": named(line.in_count_per_class),
        "out_per_class": named(line.out_count_per_class),
    }


def main() -> None:
    cfg = cli_config("Count vehicles crossing a line", "configs/count/line.yaml")
    counts = count_video(cfg)
    out = Path(cfg.get("report_path", "reports/counts.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
