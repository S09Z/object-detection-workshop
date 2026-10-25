# License register

Record the license of every dataset and model **before** downloading or using it (CLAUDE.md §5, D-010).
A tag on a Hugging Face / Kaggle mirror does not override the original dataset's license: record both.
`Commercial OK?` = may a model trained on it be used in a paid product. **Verified** = the source page was
read by a person; entries below come from the project config and need that check before phase-3-b.

## Datasets

| Name (key in config) | Source | License (original) | Mirror license | Commercial OK? | Verified | Used in | Notes |
|---|---|---|---|---|---|---|---|
| `coco` | COCO 2017 vehicle subset | Annotations CC BY 4.0; images carry their Flickr licenses | n/a | Yes, with attribution (annotations); check image terms | No | phase-3-b, 5 | Seed for car/bus/truck/motorcycle/bicycle. `truck` mixes pickups, so relabel a sample |
| `mio_tcd` | MIO-TCD (traffic cameras) | CC BY-NC-SA 4.0 | n/a | **No** (learning only) | No | phase-3-b, 4, 5 (workshop weights only) | Keep weights trained on it separate from the commercial set (D-010) |
| `roboflow_thai` | Roboflow Universe Thai vehicles (`workspace`/`project` still `CHANGE_ME`) | CC BY 4.0 | n/a | Yes, with attribution | No | phase-3-b, 5 | Tiny (~59 images); Songthaew, Taxi, Pickup, Motorcycle |
| `visaitech_mixed` | Hugging Face `visaitech/vehicle-mixed-traffic-detection` @ `de35185` (v0.1, Visaitech 2026; recorded by Visaitech, i.e. the HF repo is the original, not a mirror) | CC BY 4.0: commercial use allowed with attribution to Visaitech | same (no separate mirror) | Yes, with attribution | **Yes**, dataset card read 2026-10-20 | phase-3-b (downloaded; not in `source_mappings`) | 293 images / 1,506 boxes, 4 coarse classes (pedestrian, 2-wheeler, 3-wheeler, 4-wheeler), daylight dashcam from 6 videos. Plates, heads pixelated. Excluded the 12 GB unannotated `videos/` folder (not needed). Attribution and the card's BibTeX must go in the dataset card (phase-9-b) |
| `ua_detrac_hf_mirror` | HF `abhineet123/ua_detrac` | UA-DETRAC terms unverified | mirror tag CC-BY-4.0 | **No** until the original license is confirmed | No | disabled | Learning and benchmarking only |
| `kaggle_example` | placeholder | CHECK_ON_KAGGLE | n/a | **No** | n/a | disabled | Fill in the real slug and license before enabling |
| Freely licensed traffic clips (Pexels / Pixabay / CC BY) | `data/samples/` | per clip | n/a | Per clip | No | phase-3-b, 4-e, 6 | Record URL and license per clip in `data/samples/SOURCES.md` when added |
| Own Thai footage | cameras / partner | owned or contract | n/a | Yes, if rights and PDPA are cleared | n/a | phase-3-f | Needs `docs/privacy.md` review first |

Known non-commercial or restricted datasets that must not enter a commercial model: MIO-TCD, KITTI,
Cityscapes, Waymo (models trained on it are also restricted), DAWN, BDD100K (members only).

## Models and key libraries

| Name | License | Commercial OK? | Used for | Notes |
|---|---|---|---|---|
| RF-DETR Nano / Small / Medium (`rfdetr`) | Apache-2.0 | Yes | Detector (D-001) | Larger RF-DETR sizes may carry a different license; check before using them |
| Ultralytics YOLO (incl. fine-tuned weights) | AGPL-3.0 | Only under AGPL or an Enterprise license | Optional baseline (`yolo` extra) | Do not ship in the commercial product without a license |
| Pretrained COCO weights for RF-DETR | Follows the RF-DETR release | Check on download | phase-4-a baseline | Verify at download time |
| supervision, trackers (ByteTrack) | MIT / Apache-2.0 (verify in `uv.lock` metadata) | Yes | Metrics, tracking, line counting | |
| ONNX Runtime, TensorRT | MIT; NVIDIA license | Yes (TensorRT under NVIDIA terms) | Inference | Build engines on the target device |

## Verification checklist (before phase-3-b downloads)

- [ ] Open each dataset page and copy the license text exactly; fill the `Verified` column.
- [x] `visaitech_mixed`: license read and revision pinned (2026-10-20).
- [ ] Confirm the original license for any other source or mirror.
- [ ] Pin `revision` in `configs/data/sources.yaml` for each remaining Hugging Face source.
- [ ] Add per-clip license rows for clips placed in `data/samples/`.
