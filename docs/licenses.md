# License register

Record the license of every dataset and model **before** downloading or using it (CLAUDE.md §5, D-010).
A tag on a Hugging Face / Kaggle mirror does not override the original dataset's license: record both.
`Commercial OK?` = may a model trained on it be used in a paid product. **Verified** = the source page was
read by a person; entries below come from the project config and need that check before phase-3-b.

## Datasets

|  | Roboflow Universe  v1 (author: classification-chickhen) | CC BY 4.0 (shown on the Universe page, read 2026-10-22) | n/a | Yes, with attribution to classification-chickhen | **Yes**, page read 2026-10-22 | phase-3-b, 5 | 93 images; classes Car, truck, bus, motorcycle, van, taxi (mapped to car), pickup truck, songtaew, Tuktuk. Box counts per class unknown until downloaded. Attribution must go in the dataset card (phase-9-b) |
|---|---|---|---|---|---|---|---|
| `coco` | COCO 2017 train, vehicle subset via FiftyOne zoo (`vd-download --only coco`) | Annotations and website: CC BY 4.0. Images: Flickr Terms of Use; the COCO Consortium does not own the copyright and users accept responsibility for image use | n/a | Annotations yes, with attribution. **Images: your responsibility** (Flickr terms, per-image licenses) | **Yes**, terms page read 2026-10-20 | phase-3-b (downloaded), 4, 5 | 3,000 images containing car/bus/truck/motorcycle/bicycle (seed 42), 32,152 boxes (other classes dropped by mapping): car 6,892, bus 902, truck 1,553, motorcycle 1,427, bicycle 1,108. `truck` mixes pickups, so relabel a sample |
| `mio_tcd` | MIO-TCD Localization (`tcd.miovision.com`, 3.74 GB tar) | CC BY-NC-SA 4.0; cite Luo et al., "MIO-TCD: A new benchmark dataset for vehicle classification and localization", IEEE TIP 2018 | n/a | **No** (learning only) | **Yes**, dataset page read 2026-10-20 | phase-3-b (downloaded), 4, 5 (workshop weights only) | Subset of 5,000 images / 16,941 boxes (seed 42; half favour pickup_truck and work_van): pickup 2,886, van 591 after mapping. Share-alike: derived data must stay CC BY-NC-SA. Keep weights trained on it separate from the commercial set (D-010) |
| `roboflow_thai` | Roboflow Universe Thai vehicles (`workspace`/`project` still `CHANGE_ME`) | CC BY 4.0 | n/a | Yes, with attribution | No | phase-3-b, 5 | Tiny (~59 images); Songthaew, Taxi, Pickup, Motorcycle |
| `visaitech_mixed` | Hugging Face `visaitech/vehicle-mixed-traffic-detection` @ `de35185` (v0.1, Visaitech 2026; recorded by Visaitech, i.e. the HF repo is the original, not a mirror) | CC BY 4.0: commercial use allowed with attribution to Visaitech | same (no separate mirror) | Yes, with attribution | **Yes**, dataset card read 2026-10-20 | phase-3-b (downloaded; not in `source_mappings`) | 293 images / 1,506 boxes, 4 coarse classes (pedestrian, 2-wheeler, 3-wheeler, 4-wheeler), daylight dashcam from 6 videos. Plates, heads pixelated. Excluded the 12 GB unannotated `videos/` folder (not needed). Attribution and the card's BibTeX must go in the dataset card (phase-9-b) |
| `ua_detrac_hf_mirror` | HF `abhineet123/ua_detrac` | UA-DETRAC terms unverified | mirror tag CC-BY-4.0 | **No** until the original license is confirmed | No | disabled | Learning and benchmarking only |
| `kaggle_example` | placeholder | CHECK_ON_KAGGLE | n/a | **No** | n/a | disabled | Fill in the real slug and license before enabling |
| Pixabay traffic clips (9 clips, 305 s, Bangkok) | `data/samples/`, listed in `data/samples/SOURCES.md` | Pixabay Content License: free use, no attribution, no standalone redistribution | n/a | **Bench / demo only.** ToS prohibit data mining for machine learning purposes and bulk copying; do not train, label or publish with them | **Yes**, license summary and ToS read 2026-10-28 | phase-3-b, 4-e, 6 | Hand-downloaded. Not CC BY. Other Pexels / CC BY clips may still be added later |
| Own Thai footage | cameras / partner | owned or contract | n/a | Yes, if rights and PDPA are cleared | n/a | phase-3-f | Needs `docs/privacy.md` review first |

Known non-commercial or restricted datasets that must not enter a commercial model: MIO-TCD, KITTI,
Cityscapes, Waymo (models trained on it are also restricted), DAWN, BDD100K (members only).

## Models and key libraries

| Name | License | Commercial OK? | Used for | Notes |
|---|---|---|---|---|
| RF-DETR Nano / Small / Medium (`rfdetr`) | Apache-2.0 | Yes | Detector (D-001) | Larger RF-DETR sizes may carry a different license; check before using them |
| Ultralytics YOLO (incl. fine-tuned weights) | AGPL-3.0 | Only under AGPL or an Enterprise license | Optional baseline (`yolo` extra) | Do not ship in the commercial product without a license |
| Pretrained COCO weights for RF-DETR (`rf-detr-small.pth`) | Follows the RF-DETR release (Apache-2.0 code; the weights' own license is not stated separately on the download page) | Yes, on the same terms as the code; confirm before shipping | phase-4-a baseline; fine-tune init for the Windows smoke run | Downloaded 2026-10-29 by the user's approval from `storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth` (~386 MB) via `rfdetr` itself, cached in `~/.roboflow/models/`; a copy in the repo root is git-ignored (`*.pth`) |
| supervision, trackers (ByteTrack) | MIT / Apache-2.0 (verify in `uv.lock` metadata) | Yes | Metrics, tracking, line counting | |
| ONNX Runtime, TensorRT | MIT; NVIDIA license | Yes (TensorRT under NVIDIA terms) | Inference | Build engines on the target device |

## Verification checklist (before phase-3-b downloads)

- [ ] Open each dataset page and copy the license text exactly; fill the `Verified` column.
- [x] `visaitech_mixed`: license read and revision pinned (2026-10-20).
- [ ] Confirm the original license for any other source or mirror.
- [ ] Pin `revision` in `configs/data/sources.yaml` for each remaining Hugging Face source.
- [ ] Add per-clip license rows for clips placed in `data/samples/`.
