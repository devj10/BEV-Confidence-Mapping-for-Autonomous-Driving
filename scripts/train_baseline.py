#!/usr/bin/env python3
"""
train_baseline.py

Fine-tune YOLOv8 on the nuScenes clear-weather YOLO dataset.

Usage:
    python scripts/train_baseline.py \
        --data   /path/to/yolo_out/dataset.yaml \
        [--model yolov8m.pt] \
        [--epochs 50] \
        [--batch  16] \
        [--imgsz  640] \
        [--project runs/baseline] \
        [--name   nuscenes_clear]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on nuScenes"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to dataset.yaml produced by nuscenes_to_yolo.py",
    )
    parser.add_argument(
        "--model", default="yolov8m.pt",
        help="YOLO checkpoint to start from — pretrained COCO weight or a local .pt "
             "(default: yolov8m.pt)",
    )
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--batch",    type=int,   default=16,
                        help="Batch size per GPU (default: 16; use -1 for auto-batch)")
    parser.add_argument("--imgsz",    type=int,   default=640,
                        help="Training image size in pixels (default: 640)")
    parser.add_argument("--workers",  type=int,   default=8)
    parser.add_argument("--device",   default=None,
                        help="Device string: '0', 'mps', 'cpu' (default: auto-detect)")
    parser.add_argument("--project",  default="runs/baseline",
                        help="Root directory for saving runs (default: runs/baseline)")
    parser.add_argument("--name",     default="nuscenes_clear",
                        help="Sub-directory name for this run (default: nuscenes_clear)")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from the last checkpoint in --project/--name")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Train from random weights instead of COCO pretrained")
    parser.set_defaults(pretrained=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found: {data_path}\n"
            "Run data/nuscenes_to_yolo.py --clear-only first."
        )

    if args.resume:
        # ultralytics resume: pass the last weights path
        last_weights = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_weights.exists():
            raise FileNotFoundError(
                f"--resume requested but no checkpoint found at {last_weights}"
            )
        model = YOLO(str(last_weights))
    else:
        model = YOLO(args.model)

    train_kwargs = dict(
        data      = str(data_path),
        epochs    = args.epochs,
        batch     = args.batch,
        imgsz     = args.imgsz,
        workers   = args.workers,
        project   = args.project,
        name      = args.name,
        pretrained= args.pretrained,
        resume    = args.resume,
        # Augmentation — moderate defaults suitable for driving data
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        degrees   = 0.0,    # no rotation: cars don't appear upside-down
        translate = 0.1,
        scale     = 0.5,
        fliplr    = 0.5,
        mosaic    = 1.0,
        # Optimizer
        optimizer = "AdamW",
        lr0       = 1e-3,
        lrf       = 0.01,   # final lr = lr0 * lrf
        weight_decay = 5e-4,
        warmup_epochs= 3,
        # Logging
        plots     = True,
        save      = True,
        save_period = 10,   # checkpoint every N epochs
        val       = True,
    )

    if args.device is not None:
        train_kwargs["device"] = args.device
    else:
        # Auto-select MPS on Apple Silicon when no device is specified
        import torch
        if torch.backends.mps.is_available():
            train_kwargs["device"] = "mps"

    print(f"Model  : {args.model}")
    print(f"Data   : {data_path}")
    print(f"Epochs : {args.epochs}  |  Batch: {args.batch}  |  Imgsz: {args.imgsz}")
    print(f"Output : {args.project}/{args.name}")
    print()

    results = model.train(**train_kwargs)

    # ── Quick val on the best checkpoint ──────────────────────────────────────
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\nRunning val with best checkpoint: {best_weights}")
        best_model = YOLO(str(best_weights))
        best_model.val(data=str(data_path), imgsz=args.imgsz, batch=args.batch)

    print("\nDone. Artifacts saved to:", Path(args.project) / args.name)


if __name__ == "__main__":
    main()
