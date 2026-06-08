"""
scripts/oversample_rare.py

Duplicates training images that contain rare classes so the model sees
them more often. Run this once on the YOLO dataset before training.

Rare classes (barrier=7, traffic_cone=8, motorcycle=3, bicycle=4):
  - Each image containing one of these is copied N times.

Usage:
    python scripts/oversample_rare.py --data data/yolo_out/dataset.yaml --factor 3
"""

import argparse
import shutil
from pathlib import Path

import yaml

# Classes to oversample (indices from class_map.py)
RARE_CLASSES = {3, 4, 7, 8}  # motorcycle, bicycle, barrier, traffic_cone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True, help="Path to dataset.yaml")
    p.add_argument("--factor", type=int, default=3,
                   help="How many extra copies to add for each rare-class image (default: 3)")
    return p.parse_args()


def image_has_rare_class(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cls_id = int(line.split()[0])
        if cls_id in RARE_CLASSES:
            return True
    return False


def main():
    args = parse_args()

    with open(args.data) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg["path"])
    train_images_dir = dataset_root / cfg["train"]   # e.g. images/train
    train_labels_dir = Path(str(train_images_dir).replace("images", "labels"))

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [p for p in train_images_dir.iterdir()
                  if p.suffix.lower() in image_exts]

    rare_images = []
    for img_path in all_images:
        label_path = train_labels_dir / img_path.with_suffix(".txt").name
        if image_has_rare_class(label_path):
            rare_images.append(img_path)

    print(f"Found {len(rare_images)} rare-class images out of {len(all_images)} total.")
    print(f"Adding {args.factor} copies each → +{len(rare_images) * args.factor} images.")

    added = 0
    for img_path in rare_images:
        label_path = train_labels_dir / img_path.with_suffix(".txt").name
        for i in range(1, args.factor + 1):
            new_stem = f"{img_path.stem}_over{i}"
            new_img = train_images_dir / f"{new_stem}{img_path.suffix}"
            new_lbl = train_labels_dir / f"{new_stem}.txt"
            if not new_img.exists():
                shutil.copy(img_path, new_img)
                if label_path.exists():
                    shutil.copy(label_path, new_lbl)
                added += 1

    print(f"Done. Added {added} oversampled images to {train_images_dir}")


if __name__ == "__main__":
    main()
