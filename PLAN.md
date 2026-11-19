# PLAN.md — vehicle-detect

**เป้าหมาย:** โมเดลตรวจจับและจำแนกประเภทรถ 9 class ที่ทำงาน **≥ 60 FPS end-to-end** (p99 ≤ 16.7 ms)
บนอุปกรณ์จริง พร้อม ML lifecycle ครบ และ use case ธุรกิจที่วัดผลได้ (default: นับรถแยกประเภทที่ทางแยก)

**วิธีใช้ไฟล์นี้**
- ทำทีละ sub-task ตามลำดับ (ID: `phase-N-x`) ข้ามได้เฉพาะเมื่อเขียนเหตุผลไว้ใน `PROGRESS.md`
- **Verify** คือนิยามของคำว่าเสร็จ ติ๊ก `[x]` ได้ก็ต่อเมื่อรัน Verify แล้วผ่านเท่านั้น
- ห้ามแก้ Goal/Verify เพื่อให้ผ่าน ถ้าเกณฑ์ผิดให้เสนอแก้และถามก่อน (ดู `CLAUDE.md`)
- 1 sub-task = 1 commit (`phase-N-x: ...`) + 1 entry ใน `PROGRESS.md`

**ภาพรวม**

| Phase | หัวข้อ | Lifecycle stage | Karpathy recipe |
|---|---|---|---|
| 1 | Setup & harness | Foundation | — |
| 2 | Business framing | Problem definition | — |
| 3 | Data | Data | Become one with the data |
| 4 | End-to-end skeleton | Pipeline | Skeleton + dumb baselines, overfit |
| 5 | Train & improve | Modeling | Regularize → tune |
| 6 | Real-time optimization | Optimization | Squeeze (ภายใต้งบ 60 FPS) |
| 7 | Serving & CI/CD | Deployment | — |
| 8 | Monitoring & retraining | Operations | — |
| 9 | Capstone | Business value | — |

---

## Phase 1 — Setup & harness

> ทุก session ต่อจากนี้เริ่มจาก environment ที่สะอาดและรันซ้ำได้

- [x] **phase-1-a** สร้าง environment และ `uv.lock`
  - Goal: `make setup` สำเร็จบนเครื่อง dev (Mac) ได้ torch build ที่ถูกต้อง
  - Output: `uv.lock` (commit), `.venv/`, `.env`
  - Verify: `uv run python -c "import torch, rfdetr, supervision; print(torch.__version__, torch.backends.mps.is_available() or torch.cuda.is_available())"` พิมพ์ `True`
- [x] **phase-1-b** Git + pre-commit + first commit
  - Goal: hooks ทำงาน, notebook ถูก strip output อัตโนมัติ
  - Output: initial commit
  - Verify: `uv run pre-commit run --all-files` ผ่านทั้งหมด และ `./init.sh` จบด้วย `OK`
- [x] **phase-1-c** CI บน GitHub
  - Goal: push ขึ้น GitHub แล้ว job `lint-test` เขียว
  - Verify: `gh run list --limit 1` แสดง `completed success`
- [x] **phase-1-d** DVC + remote storage
  - Goal: `make dvc-init` ให้ remote อยู่บน local disk หรือ NAS (D-015) โดย default คือ `~/dvc-store` ส่วน NAS ให้ใช้ `DVC_REMOTE=/mnt/nas/...`
  - Output: `.dvc/config` (commit) ส่วน path เฉพาะเครื่องอยู่ใน `.dvc/config.local` (ไม่ commit) และมีแผน backup ของ disk/NAS
  - Verify: `uv run dvc remote list` แสดง `storage` และ `uv run dvc push` ไฟล์ทดสอบ 1 ไฟล์แล้วเห็นใน remote
- [x] **phase-1-e** MLflow tracking
  - Goal: เลือก local (`./mlruns`) หรือ server กลาง, ตั้งใน `.env`
  - Verify: `make mlflow` เปิด UI ได้ และ log run ทดสอบ 1 run เห็นใน UI
- [x] **phase-1-f** เครื่อง train RTX 5070 (CUDA 13)
  - Goal: Windows → WSL2 Ubuntu 24.04 + NVIDIA driver ≥ 580 บน Windows แล้ว clone repo ไว้ที่ `~/` ใน WSL
  - Output: `uv sync` ได้ torch cu130 และบันทึก driver/GPU ใน `PROGRESS.md`
  - Verify: `make gpu-check` พิมพ์ `OK: CUDA kernels run on this GPU` และระบุ `sm_120`, `bf16 supported: True`

## Phase 2 — Business framing

> ตอบให้ได้ก่อนเขียนโมเดล: ใครจ่ายเงิน, วัดอะไร, ต้องแม่นแค่ไหน, รันที่ไหน

- [x] **phase-2-a** เลือก use case และ KPI ธุรกิจ
  - Goal: เลือก 1 use case (default: traffic count ที่ทางแยก) ระบุผู้ใช้, KPI ธุรกิจ และ ML metric ที่สอดคล้อง
  - Output: `docs/problem.md` เช่น "count accuracy ≥ 95% เทียบนับมือ 15 นาที, แยก 9 class"
  - Verify: `docs/problem.md` มีหัวข้อ User / KPI / ML metric / Acceptance threshold ครบ
- [x] **phase-2-b** เลือก target hardware และ latency budget
  - Goal: อุปกรณ์คือ **RTX 5070 PC (WSL2)** (D-014) ให้ระบุจำนวนกล้อง ความละเอียด (เช่น 1080p@30) และว่ารันพร้อมกับงาน train หรือไม่
  - Output: ตาราง budget ใน `docs/problem.md`: decode / detect / track / อื่นๆ รวม ≤ 16.7 ms
  - Verify: ผลรวม budget ≤ 16.7 ms และ bench ต้องวัดตอนที่ไม่มี training รันอยู่บนเครื่องเดียวกัน
- [x] **phase-2-c** Privacy (PDPA) แบบย่อ
  - Goal: data flow ว่าภาพอยู่ที่ไหน ส่งอะไรออกนอกเครื่อง เก็บนานเท่าไร
  - Output: `docs/privacy.md` (DPIA-lite) และป้ายแจ้ง CCTV ถ้าเก็บข้อมูลเอง
  - Verify: เอกสารระบุว่าส่งออกเฉพาะ counts/boxes, ไม่มีภาพหรือป้ายทะเบียนใน git/MLflow
- [x] **phase-2-d** Freeze class taxonomy
  - Goal: ยืนยัน 9 class และลำดับใน `configs/data/thai_traffic.yaml` (เปลี่ยนลำดับภายหลัง checkpoint เดิมจะใช้ไม่ได้)
  - Output: `docs/labeling_guide.md` นิยามแต่ละ class พร้อมรูปตัวอย่างและกรณีกำกวม (pickup vs truck, van vs car)
  - Verify: `make test` ผ่าน (`test_every_config_mapping_points_to_a_real_class`)

## Phase 3 — Data ("become one with the data")

- [x] **phase-3-a** License register
  - Goal: บันทึก license ทุก dataset/model ก่อนดาวน์โหลด
  - Output: `docs/licenses.md` ตาราง: name, license, commercial OK?, ใช้ใน phase ไหน
  - Verify: ทุก source ใน `source_mappings` มีแถวใน `docs/licenses.md`
- [x] **phase-3-b** ดาวน์โหลดและแปลงเป็น COCO (`make download`, D-018)
  - Goal: เปิดใช้ source ใน `configs/data/sources.yaml` โดย HF `visaitech_mixed` เปิดไว้แล้ว ส่วน Kaggle และ Roboflow ต้องใส่ slug เอง
    และ pin `revision` COCO vehicle subset + MIO-TCD หรือ UA-DETRAC (ใช้เรียน) + Roboflow Thai (CC BY 4.0) ลง `data/raw/<source>/`
    และคลิปจราจรที่ license อนุญาต (Pexels / Pixabay / CC BY) ≥ 5 นาทีลง `data/samples/` สำหรับใช้ bench และ count (D-016)
  - Output: `images/` และ `annotations.coco.json` ต่อ source โดยใส่ `camera_id` ในแต่ละ image ถ้ามี
  - Verify: ทุก source มี `data/raw/<source>/annotations.coco.json` ที่จำนวน images > 0 และมี `SOURCE.json`
    และแถวใน `docs/licenses.md` ที่ระบุทั้ง license ต้นฉบับและ license ของ mirror
- [x] **phase-3-c** Prepare + group split
  - Goal: `make prepare` สร้าง train/valid/test แบบแยกตามกล้อง
  - Output: `data/processed/coco/`, `reports/data_stats.json`
  - Verify: `make prepare` สำเร็จ, ทุก class มี box ใน train ≥ 200 (ถ้าไม่ถึงให้บันทึกไว้เป็น gap)
- [x] **phase-3-d** EDA (`notebooks/01_eda_dataset.ipynb`)
  - Goal: ดูรูปจริง ≥ 100 รูป, class balance, ขนาด box, กลางวัน/กลางคืน
  - Output: สรุป 5–10 ข้อสังเกตใน `PROGRESS.md` (เช่น "มอเตอร์ไซค์ 40% เป็น box < 32×32")
  - Verify: notebook รันจบโดยไม่ error (`uv run jupyter nbconvert --execute --to notebook --stdout notebooks/01_eda_dataset.ipynb > /dev/null`)
- [ ] **phase-3-e** Label QA (`notebooks/02_label_quality_fiftyone.ipynb`)
  - Goal: หา duplicate, near-duplicate ข้าม split และ label ผิด
  - Output: รายการ image ที่ต้องแก้ใน `docs/label_issues.md`
  - Verify: ไม่มี near-duplicate ระหว่าง train กับ test (หรือบันทึกจำนวนที่ลบออก)
- [ ] **phase-3-f** ข้อมูลไทยของตัวเอง + pre-label (เลื่อนได้ ตอนนี้ยังไม่มี footage — D-016)
  - Goal: วางแผนแหล่ง footage (กล้องตัวเอง / ลูกค้า / หน่วยงาน) หลังผ่าน `phase-2-c` แล้ว extract เฟรม, pre-label ด้วย SAM 3 / X-AnyLabeling แล้วตรวจด้วยคนใน CVAT หรือ Label Studio
  - Output: `data/raw/thai_cctv/` อย่างน้อย 1,000 รูป ครอบคลุม songthaew / tuk_tuk / pickup
  - Verify: ทุก class มี ≥ 100 box ที่คนตรวจแล้ว, เพิ่ม source ใน `source_mappings`
- [ ] **phase-3-g** Version data ด้วย DVC
  - Goal: `dvc add data/raw` แล้ว push ไป remote
  - Verify: `uv run dvc status -c` แสดงว่า in sync และ `git log` มี commit ของ `data/raw.dvc`

## Phase 4 — End-to-end skeleton + dumb baselines

> ให้ pipeline ทั้งเส้นรันได้ก่อน ยังไม่ต้องสนความแม่น

- [ ] **phase-4-a** Pretrained baseline (ไม่ train)
  - Goal: RF-DETR pretrained COCO รันบน test set โดย map car/bus/truck/motorcycle/bicycle
  - Output: ค่า mAP baseline ใน `PROGRESS.md` (ตัวเลขที่ต้องเอาชนะ)
  - Verify: มีค่า mAP50-95 ต่อ class ที่ COCO รู้จัก
- [ ] **phase-4-b** ONNX parity test
  - Goal: ยืนยันว่า `inference.postprocess` ให้ผลเท่ากับ `model.predict()` (background slot, ลำดับ class, พิกัด box)
  - Output: `tests/test_parity.py` (mark `slow`) เทียบบนรูปตัวอย่าง 10 รูป
  - Verify: box IoU ≥ 0.99 และ class ตรงกันทุก detection ที่ conf ≥ 0.5
- [ ] **phase-4-c** Overfit one batch
  - Goal: train บน 16 รูป ให้ loss ลดเกือบ 0 (ถ้าทำไม่ได้แปลว่ามี bug ใน data/label)
  - Output: MLflow run `overfit-16`
  - Verify: train mAP50 บน 16 รูปนั้น ≥ 0.95
- [ ] **phase-4-d** Smoke run ทั้ง pipeline
  - Goal: `make repro` บน subset เล็ก (เช่น 500 รูป, 2 epoch) ผ่านทุก stage จนถึง `bench`
  - Output: `reports/eval.json`, `reports/bench.json` (บน Mac ได้ไม่ถึง 60 FPS ก็ได้ ยังไม่ใช่เป้า)
  - Verify: `uv run dvc repro` exit 0 จนถึง `evaluate` และ `bench` สร้าง report ได้
- [ ] **phase-4-e** Counting demo แบบหยาบ
  - Goal: `make count` บนคลิปทางแยกได้วิดีโอที่มี box + เส้นนับ
  - Output: `reports/count_demo.mp4`, `reports/counts.json`
  - Verify: เปิดวิดีโอดูด้วยตา track ID ไม่กระโดดจนผิดปกติ และนับได้ > 0 ทั้งสองทิศ
- [ ] **phase-4-f** Checkpoint, resume และ lineage (D-017)
  - Goal: train สั้นๆ (5 epoch) แล้วหยุดกลางคัน (Ctrl-C หลัง epoch 2) จากนั้น `make train-resume TRAIN_CONFIG=...`
  - Output: MLflow run ที่มี tag `git_sha`, `dvc_data_rev`, `sha256` และ artifact `model/`
  - Verify: run ที่ resume ทำต่อจาก epoch เดิม (ไม่เริ่ม epoch 0), `checkpoint_best_total.pth` โหลดได้
    และเหลือ interval archive ไม่เกิน `keep_last_checkpoints`

## Phase 5 — Train & improve

> เปลี่ยนทีละอย่าง, log ทุก run, เทียบกับ best run ก่อนหน้าเสมอ

- [ ] **phase-5-a** Fine-tune RF-DETR Small baseline
  - Goal: `make train-5070` บน dataset เต็ม (RTX 5070, bf16, effective batch 16)
  - Output: MLflow run `rfdetr-small-rtx5070`, `runs/rfdetr_small_rtx5070/checkpoint_best_total.pth`
    และบันทึก peak VRAM กับเวลาต่อ epoch ลง `PROGRESS.md`
  - Verify: test mAP50-95 (จาก `run_test`) > baseline จาก `phase-4-a` และ MLflow run มี tag `git_dirty=false`
- [ ] **phase-5-b** Error analysis (`notebooks/04_error_analysis.ipynb`)
  - Goal: หา 3 สาเหตุหลักของความผิดพลาด (class สับสน, box เล็ก, กลางคืน, occlusion)
  - Output: confusion matrix + 20 ตัวอย่างที่ผิดแย่ที่สุด สรุปใน `PROGRESS.md`
  - Verify: แต่ละสาเหตุมีตัวเลขรองรับ (เช่น "night mAP ต่ำกว่า day 0.15")
- [ ] **phase-5-c** Data-centric fix (active learning รอบที่ 1)
  - Goal: แก้ label / เพิ่มข้อมูลตรงจุดที่อ่อนจาก `phase-5-b` แล้ว retrain
  - Verify: metric ของ slice ที่อ่อนดีขึ้น โดย overall mAP ไม่ลดลง
- [ ] **phase-5-d** Augmentation & regularization
  - Goal: ลองทีละตัว (resolution, augment กลางคืน/ฝน, epochs, early stopping)
  - Verify: ตาราง run เทียบใน MLflow, เลือกค่าที่ดีที่สุดใส่ `configs/train/`
- [ ] **phase-5-e** เทียบขนาดโมเดล
  - Goal: Nano vs Small (vs Medium) และ YOLO26 ถ้าต้องการ (ระวัง AGPL) ทั้ง accuracy และ latency
  - Output: ตาราง accuracy–latency ใน `docs/model_selection.md`
  - Verify: เลือกโมเดลที่ mAP สูงสุดในบรรดาตัวที่ผ่าน latency budget จาก `phase-2-b`
- [ ] **phase-5-f** Model registry
  - Goal: `make register RUN_ID=<id>` ให้ได้ alias `candidate` ใน MLflow Model Registry (D-017)
  - Output: `docs/model_card.md` (data, metrics ต่อ class/slice, ข้อจำกัด, license)
  - Verify: `mlflow models` / UI แสดง version + alias `candidate`

## Phase 6 — Real-time optimization (เป้า 60 FPS)

> ทำทั้งหมดบน target device เท่านั้น

- [ ] **phase-6-a** TensorRT FP16 บน target device
  - Goal: export `format: tensorrt`, `fp16: true` บน RTX 5070 ใน WSL2 (D-014) ต้อง build engine บนเครื่องนี้เท่านั้น
  - Verify: model-only latency (`trtexec` หรือ `rfdetr` benchmark) ≤ 50% ของ budget detect ใน `phase-2-b`
- [ ] **phase-6-b** End-to-end bench รอบแรก
  - Goal: `make bench` บนคลิป ≥ 5 นาที
  - Output: `reports/bench.json` พร้อม stage breakdown (decode/detect/track)
  - Verify: บันทึกผลจริงลง `PROGRESS.md` ไม่ว่าจะผ่านหรือไม่ และระบุ stage ที่ช้าที่สุด
- [ ] **phase-6-c** แก้คอขวดของ pipeline
  - Goal: แก้ทีละจุดตามข้อมูลจาก `phase-6-b`: threaded decode, NVDEC/GStreamer, pre-processing บน GPU หรือย้ายไป DeepStream
  - Verify: `make bench` ได้ **fps ≥ 60 และ p99 ≤ 16.7 ms** (`passed: true`)
- [ ] **phase-6-d** (optional) INT8
  - Goal: ลอง INT8 PTQ เฉพาะถ้า FP16 ยังไม่ผ่าน หรือต้องรองรับหลายกล้อง
  - Verify: mAP ลดไม่เกิน 1.5 จุดเทียบ FP16 และ FPS ดีขึ้นจริง ถ้าไม่คุ้มให้บันทึกว่าไม่ใช้
- [ ] **phase-6-e** Tune tracker + เส้นนับ
  - Goal: ปรับ `ByteTrackTracker` (activation threshold, lost buffer) และ `minimum_crossing_threshold`
  - Verify: count accuracy ≥ KPI จาก `phase-2-a` เทียบนับมือ 15 นาที และ `make bench` ยังผ่าน

## Phase 7 — Serving & CI/CD

- [ ] **phase-7-a** Inference service
  - Goal: `make serve` บน target device ให้ `/detect` และ `/health` คืนเฉพาะ boxes/classes
  - Verify: `curl -F file=@sample.jpg localhost:8000/detect` ได้ JSON และไม่มีไฟล์ภาพถูกเขียนลง disk
- [ ] **phase-7-b** Docker image
  - Goal: `docker build -f docker/Dockerfile .` บน GPU server (Jetson ใช้ base L4T)
  - Verify: container รันแล้ว `/health` แสดง provider เป็น TensorRT หรือ CUDA
- [ ] **phase-7-c** GPU bench ใน CI
  - Goal: ตั้ง self-hosted runner (label `gpu`) บน RTX 5070 PC ซึ่งเข้าถึง DVC remote บน disk หรือ NAS ได้ (D-015) ให้ job `bench-gpu` ทำงานเมื่อติด label `bench`
  - Verify: PR ทดสอบที่ติด label `bench` ได้ job สีเขียว และ fail จริงเมื่อจงใจทำให้ช้า
- [ ] **phase-7-d** Promotion rule
  - Goal: เลื่อน alias `candidate` → `production` ได้ก็ต่อเมื่อ eval ≥ เกณฑ์ และ bench ผ่าน
  - Output: ขั้นตอนใน `docs/release.md`
  - Verify: เลื่อนจริง 1 ครั้งตามขั้นตอน มีหลักฐานใน MLflow

## Phase 8 — Monitoring & retraining loop

- [ ] **phase-8-a** Runtime metrics
  - Goal: export FPS, p99, จำนวน detection, ค่า confidence เฉลี่ยต่อ class ไป Prometheus แล้วดูใน Grafana
  - Verify: dashboard แสดงข้อมูลสดจากกล้องทดสอบ ≥ 1 ชั่วโมง
- [ ] **phase-8-b** Drift detection
  - Goal: เทียบ embedding/confidence ของภาพ production กับ train ด้วย Evidently (env แยกจาก FiftyOne)
  - Verify: รายงาน drift รายวัน และทดลองป้อนคลิปกลางคืนหรือฝนแล้วเห็น drift ขึ้นจริง
- [ ] **phase-8-c** Feedback queue
  - Goal: เก็บเฟรมที่ confidence ต่ำหรือ drift สูง (ผ่านเงื่อนไข PDPA และ blur หน้าและป้าย) ส่งเข้า labeling
  - Verify: มี batch ตัวอย่าง ≥ 100 เฟรมเข้า CVAT/Label Studio ได้
- [ ] **phase-8-d** Retrain รอบที่ 2
  - Goal: label batch จาก `phase-8-c` → `dvc repro` → เทียบกับ `production`
  - Verify: โมเดลใหม่ดีกว่าบน slice ที่ drift และยังผ่าน `make bench`

## Phase 9 — Capstone

- [ ] **phase-9-a** Live demo
  - Goal: demo สดบน target device: video → box + class + count แบบ real-time
  - Verify: แสดง `reports/bench.json` ที่ `passed: true` ระหว่าง demo
- [ ] **phase-9-b** เอกสารส่งมอบ
  - Output: dataset card, model card, `docs/privacy.md`, `docs/licenses.md` ที่อัปเดตแล้ว
  - Verify: ทุกเอกสารอ้างอิง MLflow run / DVC rev ที่ใช้จริง
- [ ] **phase-9-c** ROI memo
  - Goal: ต้นทุนต่อกล้อง (hardware + ไฟ + ดูแล) เทียบมูลค่าที่ได้ เช่น ทดแทนการนับมือ หรือข้อมูลสำหรับวางแผนจราจร
  - Output: `docs/roi.md` 1 หน้า
  - Verify: มีตัวเลขต้นทุนต่อกล้องต่อเดือน และ break-even
- [ ] **phase-9-d** Retrospective
  - Goal: สิ่งที่ได้ผล / ไม่ได้ผล / ทำต่อ
  - Verify: entry สุดท้ายใน `PROGRESS.md`
