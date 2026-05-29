# cs231n
Bird's-Eye-View (BEV) Confidence Mapping for Autonomous Driving

## Getting mAP Scores — Quickstart

### Prerequisites

Install dependencies via conda:

```bash
conda env create -f environment.yml
conda activate cs231n
```

---

### Step 1 — Convert nuScenes to YOLO format

Run this once to convert the raw nuScenes data into images/labels for training.

```bash
python data/nuscenes_to_yolo.py \
    --dataroot data/v1.0-mini \
    --version  v1.0-mini \
    --output   data/yolo_out \
    --val-fraction 0.15 \
    --clear-only
```

This creates `data/yolo_out/` with `images/train`, `images/val`, `labels/train`, `labels/val`, and `dataset.yaml`.

---

### Step 2 — Train the baseline

Fine-tune YOLOv8n on the nuScenes clear-weather split:

```bash
python scripts/train_baseline.py \
    --data    data/yolo_out/dataset.yaml \
    --model   yolov8n.pt \
    --epochs  20 \
    --batch   8 \
    --imgsz   640 \
    --device  cpu \
    --project runs/baseline \
    --name    nuscenes_mini
```

> **Note:** MPS (Apple Silicon) has a known bug with this version of Ultralytics — use `--device cpu`.

Checkpoints are saved to `runs/detect/runs/baseline/nuscenes_mini/weights/`. Training runs a val pass automatically at the end and saves `best.pt`.

---

### Step 3 — Evaluate and get mAP scores

```bash
python eval/detection_metrics.py \
    --weights runs/detect/runs/baseline/nuscenes_mini/weights/best.pt \
    --data    data/yolo_out/dataset.yaml \
    --save-json results/baseline_metrics.json
```

This prints per-class and overall mAP@50 / mAP@50-95 to stdout and saves a JSON file.

**Example output:**
```
  Class                  P       R     mAP@50  mAP@50-95
  car                    —       —      0.XXX      0.XXX
  truck                  —       —      0.XXX      0.XXX
  bus                    —       —      0.XXX      0.XXX
  motorcycle             —       —      0.XXX      0.XXX
  bicycle                —       —      0.XXX      0.XXX
  emergency              —       —      0.XXX      0.XXX
  pedestrian             —       —      0.XXX      0.XXX
  barrier                —       —      0.XXX      0.XXX
  traffic_cone           —       —      0.XXX      0.XXX
  all                 0.XXX   0.XXX    0.XXX      0.XXX
```

---

### Dataset

- **Source:** [nuScenes v1.0-mini](https://www.nuscenes.org/) — 10 scenes, ~400 samples, 6 cameras each
- **After clear-weather filter:** 7 scenes (3 removed for rain/night)
- **Split:** ~1,452 train images / 246 val images
- **Classes:** car, truck, bus, motorcycle, bicycle, emergency, pedestrian, barrier, traffic_cone
