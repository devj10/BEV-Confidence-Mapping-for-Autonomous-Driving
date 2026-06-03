# scripts/train_augmented.py
"""
Longer training run with:
  - Ultralytics built-in augmentations
  - Albumentations weather pipeline (optional, --weather flag)
  - DropBlock active during training (via inject_dropblock)
  - Early stopping (patience=15)
  - W&B logging
"""
import argparse
import sys
from pathlib import Path

import torch
import wandb
import yaml
from ultralytics import YOLO

# Add repo root to path so inject_dropblock is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from inject_dropblock import inject_dropblock, remove_dropblock


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/yolo_out/dataset.yaml")
    p.add_argument("--model",   default="yolov8n.pt")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--device",  default="cpu")
    p.add_argument("--patience",type=int, default=15)
    p.add_argument("--project", default="runs/augmented")
    p.add_argument("--name",    default="nuscenes_aug_dropblock")
    p.add_argument("--weather", action="store_true",
                   help="Use weather-augmented YOLO data split (requires --data to point at weather dataset.yaml)")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Load aug config
    aug_cfg = yaml.safe_load(open("configs/augmentation.yaml"))
    ult_aug = aug_cfg["ultralytics"]

    # Init W&B
    if not args.no_wandb:
        wandb.init(
            project="cs231n-bev",
            name=args.name,
            config=vars(args),
        )

    model = YOLO(args.model)

    # Inject DropBlock into backbone BEFORE training starts
    # Hooks are weight-key-neutral — checkpoint loads fine afterward
    inject_dropblock(model.model, block_size=7, drop_prob=0.1)
    print("DropBlock injected into backbone layers (2, 4, 6, 8)")

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            patience=args.patience,       # ← early stopping
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            # Ultralytics built-in augmentations from config:
            hsv_h=ult_aug["hsv_h"],
            hsv_s=ult_aug["hsv_s"],
            hsv_v=ult_aug["hsv_v"],
            degrees=ult_aug["degrees"],
            translate=ult_aug["translate"],
            scale=ult_aug["scale"],
            fliplr=ult_aug["fliplr"],
            mosaic=ult_aug["mosaic"],
            mixup=ult_aug["mixup"],
        )
    finally:
        remove_dropblock(model.model)

    # Copy best.pt to fixed export path from configs/default.yaml
    default_cfg = yaml.safe_load(open("configs/default.yaml"))
    export_path = Path(default_cfg["checkpoint"]["export_path"])
    export_path.parent.mkdir(parents=True, exist_ok=True)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    # Ultralytics saves under runs/detect/ even with custom project sometimes
    if not best.exists():
        best = Path("runs/detect") / args.project / args.name / "weights" / "best.pt"

    import shutil
    shutil.copy(best, export_path)
    print(f"\nCheckpoint exported → {export_path}")

    if not args.no_wandb:
        wandb.save(str(export_path))
        wandb.finish()


if __name__ == "__main__":
    main()