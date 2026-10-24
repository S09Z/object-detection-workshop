# HANDOFF.md — ทำงานต่อบน Windows (RTX 5070 + WSL2)

อัปเดตล่าสุด: 2026-10-21 · branch `development` @ `91ba2fa` · repo `S09Z/object-detection-workshop`

เอกสารนี้ใช้ย้ายงานจาก MacBook (dev) ไปเครื่อง train/target (Windows + RTX 5070 12 GB, WSL2 Ubuntu 24.04)
ลำดับการอ่านของ session ใหม่ยังเป็นไปตาม `CLAUDE.md` §1 (PROGRESS.md → git log → DECISIONS.md → PLAN.md → `./init.sh`)

---

## 1. สถานะตอนนี้

| หัวข้อ | สถานะ |
|---|---|
| Sub-task ที่ทำเสร็จ | 10 จาก 48: phase-1-a…1-e, 2-a…2-d, 3-a |
| Sub-task ที่ค้างอยู่ (ถัดไป) | **phase-1-f** (ต้องทำบนเครื่องนี้) แล้วตามด้วย phase-3-b ที่เหลือ |
| phase-3-b | partial: ดาวน์โหลด `visaitech_mixed`, `coco` (3,000 รูป), `mio_tcd` (5,000 รูป) แล้ว ยังขาด Roboflow Thai และคลิปทดสอบ ≥ 5 นาทีใน `data/samples/` |
| phase-3-c | ลองรัน `make prepare` แล้วแต่ยังไม่ติ๊ก: train 6,400 / valid 800 / test 800 รูป, **tuk_tuk = 0 และ songthaew = 0** ใน split ทั้งหมด |
| Tests | `uv run pytest -m "not gpu"` → 28 passed, ruff สะอาด (บน Mac) |
| ยังไม่เคยรันบน GPU | training configs, `gpu-check`, TensorRT, bench ทั้งหมด |

Class taxonomy ถูก freeze แล้ว (D-003 Accepted): `car, pickup, van, bus, truck, motorcycle, bicycle, tuk_tuk, songthaew`
ห้ามเปลี่ยนลำดับ/จำนวน เพราะ checkpoint ทุกตัวจะใช้ไม่ได้

---

## 2. ขั้นตอนตั้งเครื่อง Windows → WSL2 (= phase-1-f)

1. ติดตั้ง NVIDIA driver **≥ 580** บน **Windows เท่านั้น** (ห้ามติดตั้ง driver ใน WSL)
2. เปิด PowerShell (admin): `wsl --install -d Ubuntu-24.04` แล้ว restart
3. ใน Ubuntu: `nvidia-smi` ต้องเห็น RTX 5070 และ driver ≥ 580
4. ติดตั้งเครื่องมือ:

```bash
sudo apt update && sudo apt install -y build-essential git make curl libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/install.sh | sh        # แล้วเปิด shell ใหม่
```

5. Clone ไว้ใน `~/` ของ WSL (**ไม่ใช่** `/mnt/c/...` เพราะ I/O ช้ามาก):

```bash
cd ~ && git clone https://github.com/S09Z/object-detection-workshop.git
cd object-detection-workshop && git switch development
make setup          # uv sync --all-groups + pre-commit + nbstripout + .env
make gpu-check      # ต้องพิมพ์: OK: CUDA kernels run on this GPU (sm_120), bf16 supported: True
```

**Verify ของ phase-1-f:** `make gpu-check` พิมพ์ `OK: CUDA kernels run on this GPU` และระบุ `sm_120`, `bf16 supported: True`
จากนั้นบันทึก driver/GPU ลง `PROGRESS.md`, ติ๊ก `[x] phase-1-f` ใน `PLAN.md` (แก้เฉพาะ checkbox), commit `phase-1-f: ...`

ข้อควรระวัง:
- `uv.lock` ถูก lock จาก Mac ซึ่งรวม wheel `cu130` ของ Linux ไว้แล้ว แต่ **ยังไม่เคยติดตั้งจริงบน Linux** ถ้า `uv sync` พัง ให้แก้ที่ index ใน `pyproject.toml` และห้ามเปลี่ยนเป็น cu128/cu126 (PyTorch 2.14 ไม่มี cu128, cu126 ไม่มี kernel sm_120)
- error `no kernel image is available` = torch build ผิด ให้ `uv sync --reinstall-package torch`
- `.python-version` = 3.13 (uv จะโหลดให้เอง)
- ห้ามรัน `make bench` ตอนที่มี training ใช้ GPU เดียวกันอยู่ ผลจะใช้ไม่ได้

---

## 3. ข้อมูล: อะไรย้ายตาม git ได้ และอะไรต้องทำใหม่

Git ไม่เก็บ `data/`, `models/`, `runs/`, `mlflow.db`, `.env` (อยู่ใน `.gitignore`) และ **ข้อมูลยังไม่ได้ `dvc add`**
(phase-3-g ยังไม่ได้ทำ) จึงยังไม่มีอะไรใน DVC remote นอกจากไฟล์ทดสอบที่ลบแล้ว

วิธีได้ข้อมูลบน Windows (เลือก 1 ทาง):

| ทาง | คำสั่ง | หมายเหตุ |
|---|---|---|
| ดาวน์โหลดซ้ำจาก config (แนะนำ) | `uv run vd-download` แล้ว `uv run vd-download --only mio_tcd` | revision ของ visaitech ถูก pin แล้ว (`de35185`), COCO และ MIO-TCD ใช้ seed 42 จึงได้ subset เดิม; COCO ผ่าน FiftyOne ~1.4 GB, MIO-TCD 3.7 GB (cache รวม ~7 GB ลบได้ภายหลัง) |
| คัดลอกจาก Mac | `rsync -a data/raw/ <user>@<windows>:~/object-detection-workshop/data/raw/` | `data/raw` มี 3 source รวม ~700 MB (ไม่ต้องก๊อป `data/_downloads`) |

หลังได้ `data/raw` แล้ว: `make prepare` (ผลลัพธ์ต้องใกล้เคียง 6,400 / 800 / 800 รูป)

DVC: `make dvc-init` อีกครั้งบนเครื่องนี้ (path ของ remote อยู่ใน `.dvc/config.local` ซึ่งไม่ commit)
ค่าเริ่มต้น `~/dvc-store`; ถ้ามี NAS ให้ `make dvc-init DVC_REMOTE=/mnt/nas/dvc-store` (ยังไม่ตัดสินใจเรื่อง NAS/backup)

MLflow: เครื่องนี้มี `mlflow.db` ของตัวเอง (`sqlite:///mlflow.db`) UI: `make mlflow` เปิดที่ **http://localhost:5001**
(ไม่ใช้ 5000 เพราะชนกับ AirPlay บน macOS; บน Windows เปลี่ยนได้ด้วย `MLFLOW_PORT=...`)

---

## 4. ลำดับงานถัดไป

1. **phase-1-f** (ข้อ 2 ด้านบน)
2. **phase-3-b** ส่วนที่เหลือ: Roboflow Thai (ต้องรู้ `workspace`/`project` slug ใส่ใน `sources.yaml` แล้วตั้ง `ROBOFLOW_API_KEY` ใน `.env`) และคลิปจราจร license เสรี ≥ 5 นาทีลง `data/samples/` พร้อมบันทึก URL/license ต่อคลิปใน `data/samples/SOURCES.md`
   ต้องอ่านหน้า license ของแต่ละ source ก่อนดาวน์โหลด และ **ถามผู้ใช้ก่อนดาวน์โหลดไฟล์ใหม่**
3. **phase-3-c** ติ๊กได้เมื่อทุก class มี box ใน train ≥ 200 หรือบันทึกเป็น gap (ตอนนี้ tuk_tuk/songthaew ขาด)
4. phase-3-d (EDA) → 3-e → 3-g → phase 4 (baseline, parity test, overfit one batch) → phase 5 train

การ train ครั้งแรกบน 5070: `make train-5070` (RF-DETR Small, bf16, batch 4 × accum 4) แล้วบันทึก **peak VRAM และ s/epoch** ลง `PROGRESS.md`
ถ้า OOM: `batch_size: 2` + `grad_accum_steps: 8` หรือเปิด `gradient_checkpointing`; ถ้า RAM ไม่พอใน WSL2 ลด `num_workers` เหลือ 4
อย่าเริ่ม train จริงบนข้อมูลชุดนี้: มี 4 class ที่ว่างอยู่

---

## 5. Workflow ของโปรเจกต์ (ตกลงกับผู้ใช้แล้ว)

- ทำงานบน branch `development` ในเครื่อง ไม่ commit ลง `main`
- จบแต่ละ sub-task: รัน Verify → ติ๊ก `PLAN.md` → อัปเดต `PROGRESS.md` (§1 overwrite, §2 append) → commit ชื่อ `phase-X-y: <what>` → push `development`
- **1 PR ต่อ 1 sub-task:** ตัด feature branch จาก `origin/development` แล้วเปิด **draft** PR เข้า `main` (ไม่ merge เอง)
- ข้อมูล/น้ำหนัก/engine ห้าม commit (DVC เท่านั้น); ภาพดิบ ใบหน้า ป้ายทะเบียนห้ามออกนอกเครื่อง (`docs/privacy.md`)
- ข้อที่ต้องถามผู้ใช้ก่อน: license ใหม่, ดาวน์โหลดไฟล์ใหม่, dependency หนัก, เปลี่ยน taxonomy, แก้ Verify ใน PLAN.md

### PR ที่เปิดค้างอยู่ (ทั้งหมด draft)

| PR | Branch | หมายเหตุ |
|---|---|---|
| #1, #2, #3 | `feature/phase-1-setup`, `feature/phase-1-d-dvc`, `feature/phase-1-dvc-mlflow` | ซ้ำซ้อน: #4 มี commit ทั้งหมดนี้แล้ว ปิดได้ |
| #4 | `feature/phase-2-business-framing` | phase 1-c…3-a (รวม phase 2) |
| #5, #6, #8 | `feature/phase-3-b-visaitech`, `-coco`, `-mio-tcd` | ซ้อนต่อกัน (stacked) |
| #7 | `feature/drop-crowd-boxes` | ตัด box `iscrowd=1` ใน `remap_coco` |

Merge ตามลำดับ #4 → #5 → #6 → #7 → #8 เพราะแต่ละ branch มี commit ของตัวก่อนหน้า
หลัง merge ให้ rebase `development` บน `main` เพื่อให้ PR ถัดไปแสดงเฉพาะ commit ของตัวเอง

---

## 6. ปัญหาที่รู้แล้วและยังไม่ได้แก้

จากการ review โปรเจกต์ (ยังไม่มีใน PLAN.md ต้องจัดเข้า sub-task เมื่อถึงเวลา):

| เรื่อง | รายละเอียด | ควรแก้ตอน |
|---|---|---|
| TensorRT provider | `OnnxDetector` เรียก `TensorrtExecutionProvider` โดยไม่ตั้ง `trt_fp16_enable` และ engine cache; config `format: tensorrt` สร้างไฟล์ที่ onnxruntime โหลดไม่ได้ | ก่อน phase-6-a |
| `vd-bench` | ไม่เช็กความยาววิดีโอขั้นต่ำ (กติกาคือ ≥ 5 นาที) จึงผ่านได้บนคลิปสั้น; preprocess ทำบน CPU และไม่แยกเวลาใน stage breakdown | phase-6-b |
| Postprocess / category id | `inference.postprocess` ยังไม่ตรวจกับ ONNX จริง; `prepare` เขียน category id เริ่มที่ 0 ซึ่ง RF-DETR/Roboflow อาจคาดหวังต่างกัน | phase-4-b (parity test) |
| `train.py` lineage | ใช้ `RFDETR.from_checkpoint`, `project=`, `run=` ตามที่คาดไว้ ยังไม่เคยรันจริง; ถ้าหา run ตามชื่อไม่เจอจะแค่เตือน (run นั้นจะไม่นับตาม D-017) | phase-4-f |
| Serve app | `async def` เรียก inference แบบ blocking, ไม่มี limit ขนาดไฟล์/auth, bind `0.0.0.0`, โหลดโมเดลตอน import | phase-7-a |
| Docker | ติดตั้ง torch/rfdetr ทั้งที่ serve ใช้แค่ onnxruntime; ยังไม่ยืนยันว่า `onnxruntime-gpu` ใช้กับ CUDA 13 ได้ | phase-7-b |
| Commercial split | `prepare` รวม MIO-TCD (non-commercial) กับ source CC BY ใน split เดียว น้ำหนักที่ได้จึงเป็น learning-only (D-010); เคยเริ่มทำ flag `--commercial-only` แล้วหยุด (ยังไม่มี code) | ก่อน phase-5 |
| ข้อมูล | MIO-TCD ไม่มี camera id (frame จากกล้องเดียวกันอาจอยู่คนละ split); box < 32×32 px คิดเป็น ~40% ของ train; license ทุกแถวใน `docs/licenses.md` ยัง "Verified" เฉพาะ visaitech, coco, mio_tcd | ระหว่าง phase 3 |
| README | Quick start ยังเขียน torch index เป็น `cu128` (ที่ถูกคือ `cu130`) | ครั้งหน้าที่แตะ README |

คำถามที่ยังเปิดอยู่ (ดู `PROGRESS.md` §1): path NAS และแผน backup ของ DVC, ผู้ควบคุมข้อมูล (data controller) ถ้าเก็บ footage จริง

---

## 7. ไฟล์ที่ควรอ่านก่อนเริ่ม

| ไฟล์ | เนื้อหา |
|---|---|
| `CLAUDE.md` | กฎการทำงานของ agent, session protocol, guardrails |
| `PROGRESS.md` | สถานะปัจจุบัน + log ทุก session |
| `PLAN.md` | checklist ทั้งหมดพร้อมเกณฑ์ Verify |
| `DECISIONS.md` | D-001…D-018 (ห้ามย้อน Accepted โดยไม่เขียนอันใหม่) |
| `docs/problem.md` | use case (นับรถที่ทางแยก), KPI (นับแม่น ≥ 95% รวม และ ≥ 90% ต่อ class), latency budget 15.5 ms จาก 16.7 ms |
| `docs/privacy.md`, `docs/licenses.md`, `docs/labeling_guide.md` | PDPA, license register, นิยาม 9 class |

คำสั่งทั้งหมด: `make help` · pipeline: `make prepare → train → export → eval → bench → count` หรือ `make repro`
