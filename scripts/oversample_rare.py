"""
scripts/oversample_rare.py

Generates a train.txt listing for oversampled training.
Rare-class images are repeated N extra times by listing their path multiple
times — YOLO's dataloader supports path-list files for train:.

No image copying needed, so this runs in seconds even on network volumes.

Rare classes (barrier=7, traffic_cone=8, motorcycle=3, bicycle=4):
  - Each image containing one of these appears factor+1 times total.

Usage:
    python scripts/oversample_rare.py \
        --data /root/outputs/yolo_full/dataset.yaml \
        --factor 3 \
        --out /tmp/train_oversampled.txt
"""

import argparse
import subprocess
from pathlib import Path

import yaml

# Classes to oversample (indices from class_map.py)
RARE_CLASSES = {3, 4, 7, 8}  # motorcycle, bicycle, barrier, traffic_cone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True, help="Path to dataset.yaml")
    p.add_argument("--factor", type=int, default=3,
                   help="Extra copies for each rare-class image (default: 3)")
    p.add_argument("--out",    default="/tmp/train_oversampled.txt",
                   help="Where to write the path-list txt (default: /tmp/train_oversampled.txt)")
    return p.parse_args()


def find_rare_label_files(labels_dir: Path) -> set[Path]:
    """
    Use grep to find label files that contain rare class IDs.
    grep scans files in bulk — far faster than Python opening each one on a
    network-mounted volume.
    Pattern matches lines that start with one of the rare class IDs.
    """
    # Build an ERE pattern like '^(3|4|7|8) '
    pattern = "^(" + "|".join(str(c) for c in sorted(RARE_CLASSES)) + ") "

    result = subprocess.run(
        ["bash", "-lc",
         f"find {labels_dir} -name '*.txt' -print0 "
         f"| xargs -0 grep -Ele '{pattern}' /dev/null "
         f"| cut -d: -f1 | sort -u"],
        capture_output=True,
        text=True,
    )
    return {Path(p.strip()) for p in result.stdout.splitlines() if p.strip()}


def main():
    args = parse_args()

    with open(args.data) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg["path"])
    train_images_dir = dataset_root / cfg["train"]   # e.g. images/train
    train_labels_dir = Path(str(train_images_dir).replace("images", "labels"))

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = sorted(
        p for p in train_images_dir.iterdir()
        if p.suffix.lower() in image_exts
    )

    print(f"Scanning {len(all_images)} label files for rare classes via grep...")
    rare_label_paths = find_rare_label_files(train_labels_dir)

    # Build the oversampled path list
    lines: list[str] = []
    rare_count = 0
    for img_path in all_images:
        lines.append(str(img_path))
        label_path = train_labels_dir / img_path.with_suffix(".txt").name
        if label_path in rare_label_paths:
            rare_count += 1
            for _ in range(args.factor):
                lines.append(str(img_path))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    print(f"Found {rare_count} / {len(all_images)} rare-class images.")
    print(f"Oversampled list: {len(lines)} entries → {out}")


if __name__ == "__main__":
    main()
