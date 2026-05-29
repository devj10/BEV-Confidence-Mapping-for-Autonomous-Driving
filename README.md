# cs231n
Bird's-Eye-View (BEV) Confidence Mapping for Autonomous Driving

## Getting mAP Scores — Quickstart

### Prerequisites

Install dependencies via conda:

```bash
conda env create -f environment.yml
conda activate cs231n
```

Place the nuScenes v1.0-mini dataset under `data/v1.0-mini/` (already in `.gitignore`). The directory should contain `samples/`, `sweeps/`, `maps/`, and `v1.0-mini/`.

---

### Step 1 — Convert nuScenes to YOLO format

Run once to convert raw nuScenes annotations into YOLO-format images and labels.

```bash
python data/nuscenes_to_yolo.py \
    --dataroot data/v1.0-mini \
    --version  v1.0-mini \
    --output   data/yolo_out \
    --val-fraction 0.15 \
    --clear-only
```

- `--clear-only` keeps only clear-weather daytime scenes (removes rain/night/fog)
- `--val-fraction 0.15` holds out 15% of scenes for validation

Output: `data/yolo_out/` with `images/train` (1,452 images), `images/val` (246 images), matching `labels/`, and `dataset.yaml`. This folder is gitignored.

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

> **Note:** MPS (Apple Silicon) has a known bug with this version of Ultralytics — always use `--device cpu`.

Checkpoints are saved to `runs/detect/runs/baseline/nuscenes_mini/weights/`. The best checkpoint (`best.pt`) is saved automatically based on val mAP.

Each epoch takes ~5 min on CPU (Apple M3 Pro). 20 epochs ≈ ~1.5 hours.

---

### Step 3 — Evaluate and get mAP scores

```bash
python eval/detection_metrics.py \
    --weights runs/detect/runs/baseline/nuscenes_mini/weights/best.pt \
    --data    data/yolo_out/dataset.yaml \
    --save-json results/baseline_metrics.json
```

Prints per-class and overall mAP@50 / mAP@50-95 to stdout and saves a JSON to `results/baseline_metrics.json`.

---

### Baseline Results (v1.0-mini, 17 epochs, YOLOv8n)

| Class        | mAP@50 | mAP@50-95 |
|--------------|--------|-----------|
| car          | 0.582  | 0.294     |
| truck        | 0.378  | 0.241     |
| bus          | 0.246  | 0.175     |
| motorcycle   | 0.180  | 0.063     |
| bicycle      | 0.004  | 0.003     |
| pedestrian   | 0.559  | 0.264     |
| **all**      | **0.325** | **0.173** |

> Emergency, barrier, and traffic_cone had no detections on the single val scene in v1.0-mini.

---

### Dataset

- **Source:** [nuScenes v1.0-mini](https://www.nuscenes.org/) — 10 scenes, ~400 samples, 6 cameras each
- **After clear-weather filter:** 7 scenes (3 removed for rain/night)
- **Split:** 1,452 train images / 246 val images
- **Classes (9):** car, truck, bus, motorcycle, bicycle, emergency, pedestrian, barrier, traffic_cone
- **Projection:** 3D bounding boxes projected to 2D via camera intrinsics (6 cameras per frame)
