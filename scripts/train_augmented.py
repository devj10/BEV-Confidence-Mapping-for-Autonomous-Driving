# scripts/train_augmented.py
"""
Longer training run with:
  - Ultralytics built-in augmentations
  - Optional augmentation config via --aug-config
  - Optional W&B logging
  - DropBlock active during training
  - Early stopping
  - Robust checkpoint export
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml
import wandb
from ultralytics import YOLO

# Add repo root to path so inject_dropblock is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inject_dropblock import inject_dropblock, remove_dropblock


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/yolo_out/dataset.yaml")
    p.add_argument("--model",   default="yolov8s.pt")
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
    p.add_argument("--export-path", default=None,
                   help="Where to copy best.pt after training")
    p.add_argument("--aug-config", default=None,
                   help="Path to augmentation YAML config")

    return p.parse_args()


def load_yaml_safe(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_aug_config(aug_config_path: str | None):
    """
    Returns a complete Ultralytics augmentation dict.

    If --aug-config is omitted, uses defaults.
    If file exists but is empty, uses defaults.
    If file has only some keys, fills missing keys with defaults.
    """
    aug = dict(DEFAULT_ULTRALYTICS_AUG)

    if aug_config_path is None:
        print("No --aug-config provided. Using default Ultralytics augmentations.")
        return aug

    path = Path(aug_config_path)
    cfg = load_yaml_safe(path)

    user_aug = cfg.get("ultralytics", {})
    if user_aug is None:
        user_aug = {}

    if not isinstance(user_aug, dict):
        raise ValueError(
            f"'ultralytics' in {path} must be a dict, got {type(user_aug)}"
        )

    aug.update(user_aug)
    print(f"Loaded augmentation config from {path}")
    return aug


def resolve_export_path(args) -> Path:
    """
    Priority:
      1. --export-path
      2. configs/default.yaml checkpoint.export_path
      3. model_final.pt in repo root
    """
    if args.export_path:
        return Path(args.export_path)

    default_cfg_path = REPO_ROOT / "configs" / "default.yaml"

    if default_cfg_path.exists():
        cfg = load_yaml_safe(default_cfg_path)
        checkpoint_cfg = cfg.get("checkpoint", {}) or {}
        export_path = checkpoint_cfg.get("export_path")
        if export_path:
            return Path(export_path)

    return REPO_ROOT / "results" / "checkpoints" / "model_final.pt"


def find_best_checkpoint(project: str, name: str) -> Path:
    """
    Ultralytics usually saves:
      {project}/{name}/weights/best.pt

    This function checks common locations and falls back to recursive search.
    """
    candidates = [
        Path(project) / name / "weights" / "best.pt",
        REPO_ROOT / project / name / "weights" / "best.pt",
        Path("runs") / "detect" / name / "weights" / "best.pt",
        REPO_ROOT / "runs" / "detect" / name / "weights" / "best.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(REPO_ROOT.rglob("best.pt"))
    if matches:
        # Pick most recently modified best.pt
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    matches = list(REPO_ROOT.rglob("last.pt"))
    if matches:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    raise FileNotFoundError(
        "Could not find best.pt or last.pt after training. "
        f"Checked project={project}, name={name}"
    )


def validate_dataset_yaml(data_path: str):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")

    cfg = load_yaml_safe(path)

    required = ["path", "train", "val", "names", "nc"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Dataset YAML missing required keys: {missing}")

    print("Dataset YAML loaded:")
    print(yaml.dump(cfg, sort_keys=False))


def main():
    args = parse_args()

    # Load aug config
    aug_config_path = args.aug_config or "configs/augmentation.yaml"
    aug_cfg = yaml.safe_load(open(aug_config_path))
    ult_aug = aug_cfg["ultralytics"]

    # Init W&B
    if not args.no_wandb:
        wandb.init(
            project="cs231n-bev",
            name=args.name,
            config=vars(args),
        )

    print("Loading YOLO model...")
    model = YOLO(args.model)

    # Inject DropBlock into backbone BEFORE training starts
    # Hooks are weight-key-neutral — checkpoint loads fine afterward
    inject_dropblock(model.model, block_size=7, drop_prob=0.1)
    print("DropBlock injected into backbone layers (2, 4, 6, 8)")

    try:
        print("Starting YOLO training...")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            patience=args.patience,
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
            cls=1.5,       # upweight classification loss for rare classes
            cos_lr=True,   # cosine LR schedule instead of linear decay
        )
    finally:
        print("Removing DropBlock...")
        remove_dropblock(model.model)

    best = find_best_checkpoint(args.project, args.name)
    export_path = resolve_export_path(args)

    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, export_path)

    print(f"\nBest checkpoint found at: {best}")
    print(f"Checkpoint exported → {export_path}")

    if not args.no_wandb:
        wandb.save(str(export_path))
        wandb.finish()


if __name__ == "__main__":
    main()
