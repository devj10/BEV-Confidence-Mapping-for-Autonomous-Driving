# BEV Confidence Mapping for Autonomous Driving

Uncertainty-aware Bird's-Eye-View (BEV) projection pipeline on nuScenes.
Combines MC-DropBlock inference, 3D lifting, and Gaussian splatting to produce a
per-frame confidence heat map in ego-frame meters.

---

## Pipeline overview

```
nuScenes frame
      │
      ▼
  YOLOv8n detector  (fine-tuned on nuScenes clear-weather split)
      │  2D boxes + confidence
      ▼
  MC-DropBlock  (T=20 stochastic forward passes)
      │  per-pass boxes → pixel variance (var_u)
      ▼
  lift_to_3d  ──── GT depth  (calib + ego_pose)
      │         └── LiDAR depth  (lidar_project.py → var_z)
      │  (x_m, y_m, sigma_x, sigma_y) in ego frame
      ▼
  uncertainty_to_bev  (pixel var → BEV sigma in meters)
      │  sigma_lat = (z/fx)·√var_u,  sigma_fwd = √var_z
      ▼
  lift_adapter  →  LiftedDetection objects
      ▼
  splat.py  (single_box or all_t mode)
      │  200×200 float32 heat map
      ▼
  export_bev.py  →  bev_frame.json + bev_frame.png
```

**Key finding:** uncertainty is depth-dominated. At 30 m with 3 px jitter,
`sigma_fwd ≈ 0.9 m` vs `sigma_lat ≈ 0.07 m` — a 12× ratio.
Far cars produce tall, elongated blobs; near cars produce tight dots.

---

## Module map

### Detection & MC inference

| File | What it does |
|------|-------------|
| `yolov8_backbone.py` | YOLOv8 feature extractor |
| `dropblock.py` | DropBlock layer + MC-inference toggle |
| `inject_dropblock.py` | Hooks DropBlock into a YOLOv8 backbone in-place |
| `mc_yolo.py` | Runs T stochastic forward passes; outputs per-pass JSON |

### BEV pipeline (`bev/`)

| File | What it does |
|------|-------------|
| `bev_grid.py` | Grid constants (200×200, 0.25 m/cell, x∈[0,50] m, y∈[−25,25] m) |
| `lift_to_3d.py` | Back-project 2D detections to ego-frame 3D (GT-depth + LiDAR-depth paths) |
| `lidar_project.py` | Project LiDAR points onto image; extract robust per-box depth + variance |
| `uncertainty_to_bev.py` | Convert pixel variance → BEV sigma in meters |
| `lift_adapter.py` | Batch-lift raw dicts → `LiftedDetection` objects; filter `None` failures |
| `splat.py` | 2D Gaussian splatting onto the BEV grid (single_box + all_t modes) |
| `run_bev.py` | Top-level conductor: lift pre-pass → splat → `BevFrame` |

### Uncertainty scoring (`uncertainty/`)

| File | What it does |
|------|-------------|
| `scores.py` | Center variance, box variance, confidence variance, entropy, combined score |
| `aggregate.py` | Fuse T per-pass detections into one set of clustered detections |
| `associate.py` | Match detections across passes (IoU-based) |

### Evaluation (`eval/`, `bev_uncertainty/eval/`)

| File | What it does |
|------|-------------|
| `eval/detection_metrics.py` | Standard mAP@50 / mAP@50-95 |
| `eval/mc_detection_metrics.py` | mAP after MC-DropBlock fusion |
| `eval/bev_metrics.py` | BEV-space localisation metrics |
| `bev_uncertainty/eval/calibration.py` | Uncertainty calibration (ECE, reliability diagrams) |
| `bev_uncertainty/eval/sanity_checks.py` | Grid/pipeline sanity checks |

### Scripts (`scripts/`)

| File | What it does |
|------|-------------|
| `train_baseline.py` | Fine-tune YOLOv8n on nuScenes clear-weather split |
| `train_augmented.py` | Train with data augmentation |
| `run_mc_dropblock_inference.py` | Full MC inference run on val set |
| `batch_mc_inference.py` | Batch version for multiple scenes |
| `batch_bev_evaluation.py` | Run BEV pipeline over a batch of frames |
| `validate_bev_lifting.py` | Validate GT-depth and LiDAR-depth lifting |
| `visualize_rgb_bev.py` | Matplotlib BEV overlay (GT boxes + lifted detections) |
| `export_bev.py` | Export BEV frame to JSON + PNG for Three.js frontend |
| `compute_bev_metrics.py` | Aggregate BEV metrics across scenes |

---

## Quickstart

### 0. Install

```bash
conda env create -f environment.yml
conda activate cs231n
```

Place nuScenes v1.0-mini under `data/v1.0-mini/` (gitignored).

---

### 1. Convert nuScenes → YOLO format

```bash
python data/nuscenes_to_yolo.py \
    --dataroot data/v1.0-mini \
    --version  v1.0-mini \
    --output   data/yolo_out \
    --val-fraction 0.15 \
    --clear-only
```

Output: `data/yolo_out/` — 1,452 train / 246 val images, `dataset.yaml`.

---

### 2. Train baseline detector

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

> **Apple Silicon note:** use `--device cpu`; MPS has a known bug with this Ultralytics version.

Checkpoints → `runs/detect/runs/baseline/nuscenes_mini/weights/best.pt`

---

### 3. Evaluate detection mAP

```bash
python eval/detection_metrics.py \
    --weights runs/detect/runs/baseline/nuscenes_mini/weights/best.pt \
    --data    data/yolo_out/dataset.yaml \
    --save-json results/baseline_metrics.json
```

---

### 4. MC-DropBlock inference

Run T=20 stochastic passes per frame:

```bash
python mc_yolo.py \
    --weights runs/detect/runs/baseline/nuscenes_mini/weights/best.pt \
    --source  data/yolo_out/images/val \
    --T       20 \
    --out     results/mc_raw_detections.json
```

Use `--max-images 5` for a quick smoke test.

---

### 5. Export BEV frame (stub — no nuScenes required)

```bash
python scripts/export_bev.py
# → results/bev_export/bev_frame.json   (40 000-cell grid, detection list)
# → results/bev_export/bev_frame.png    (green→red confidence heat map)
```

With a real sample token:

```bash
python scripts/export_bev.py \
    --sample-token <token> \
    --depth-mode gt \
    --mode single_box
```

---

### 6. Smoke-test the BEV pipeline

```bash
python bev/splat.py          # splatting + grid constants
python bev/run_bev.py        # full conductor (both modes)
python bev/uncertainty_to_bev.py  # sigma propagation + widening visual
```

All three scripts are self-contained and require no nuScenes data.

---

## Baseline results (v1.0-mini, 17 epochs, YOLOv8n)

| Class | mAP@50 | mAP@50-95 |
|-------|--------|-----------|
| car | 0.582 | 0.294 |
| truck | 0.378 | 0.241 |
| bus | 0.246 | 0.175 |
| motorcycle | 0.180 | 0.063 |
| bicycle | 0.004 | 0.003 |
| pedestrian | 0.559 | 0.264 |
| **all** | **0.325** | **0.173** |

> Emergency, barrier, and traffic_cone had no detections on the single val scene in v1.0-mini.

---

## BEV grid spec

| Parameter | Value |
|-----------|-------|
| x (forward) | 0 – 50 m |
| y (lateral) | −25 – 25 m |
| Cell size | 0.25 m |
| Grid shape | 200 × 200 |
| Origin | ego frame (x fwd, y left, z up) |

---

## Dataset

- **Source:** [nuScenes v1.0-mini](https://www.nuscenes.org/) — 10 scenes, ~400 samples, 6 cameras
- **After clear-weather filter:** 7 scenes (3 removed for rain / night)
- **Split:** 1,452 train / 246 val images
- **Classes (9):** car, truck, bus, motorcycle, bicycle, emergency, pedestrian, barrier, traffic_cone
