#!/usr/bin/env python3
"""
nuscenes_to_yolo.py

Convert nuScenes 3D annotations to YOLO 2D format.

Pipeline per image:
  global 3D box → ego frame → camera frame → project 8 corners → tight 2D bbox → YOLO txt

Usage:
    python nuscenes_to_yolo.py \
        --dataroot /path/to/nuscenes \
        --version  v1.0-trainval \
        --output   /path/to/yolo_dataset \
        [--val-fraction 0.1] \
        [--visualize]
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from tqdm import tqdm

# Allow running from repo root or from data/
sys.path.insert(0, str(Path(__file__).parent))
from class_map import CLASSES, NUM_CLASSES, get_class_id

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

def get_weather_pipeline():
    return A.Compose([
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.35, alpha_coef=0.1, p=0.3),
        A.RandomRain(slant_lower=-10, slant_upper=10,
                     drop_length=15, drop_width=1,
                     brightness_coefficient=0.9, p=0.3),
        A.MotionBlur(blur_limit=(3, 9), p=0.2),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))

def _project_box(box, cs_record: dict, ego_record: dict, K: np.ndarray):
    """
    Project a nuScenes Box (global frame) into image coordinates.

    Returns (corners_2d, depths_cam) where:
      corners_2d : (8, 2) float — pixel coordinates of the 8 cuboid corners
      depths_cam : (8,)   float — z-depth of each corner in camera frame

    Returns None if every corner is behind the camera (z <= 0).
    """
    box = box.copy()

    # 1. Global → ego vehicle frame
    box.translate(-np.array(ego_record["translation"]))
    box.rotate(Quaternion(ego_record["rotation"]).inverse)

    # 2. Ego → camera sensor frame
    box.translate(-np.array(cs_record["translation"]))
    box.rotate(Quaternion(cs_record["rotation"]).inverse)

    corners_3d = box.corners()          # (3, 8) in camera frame
    depths     = corners_3d[2, :]       # z per corner

    if np.all(depths <= 0):
        return None

    # 3. Perspective projection:  K @ [X; Y; Z]  then  / Z
    pts        = K @ corners_3d         # (3, 8)
    pts[:2, :] /= pts[2, :]             # divide x,y by z
    corners_2d = pts[:2, :].T           # (8, 2)

    return corners_2d, depths


def _corners_to_yolo(corners_2d: np.ndarray, img_w: int, img_h: int):
    """
    Tightest 2D box around projected corners → YOLO (cx, cy, w, h) normalized.
    Returns None if the clipped box has zero area (entirely off-screen).
    """
    x_min = np.clip(corners_2d[:, 0].min(), 0, img_w)
    x_max = np.clip(corners_2d[:, 0].max(), 0, img_w)
    y_min = np.clip(corners_2d[:, 1].min(), 0, img_h)
    y_max = np.clip(corners_2d[:, 1].max(), 0, img_h)

    if x_max <= x_min or y_max <= y_min:
        return None

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h

    return cx, cy, w, h


def process_scene(nusc: NuScenes, scene: dict, split: str,
                  output_dir: Path, visualize: bool, weather_pipeline=None) -> None:
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    vis_dir    = output_dir / "visualizations" / split

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    sample_token = scene["first_sample_token"]

    while sample_token:
        sample = nusc.get("sample", sample_token)

        for cam in CAMERAS:
            if cam not in sample["data"]:
                continue

            cam_token  = sample["data"][cam]
            cam_data   = nusc.get("sample_data", cam_token)
            cs_record  = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            ego_record = nusc.get("ego_pose",          cam_data["ego_pose_token"])
            K          = np.array(cs_record["camera_intrinsic"])

            img_w = cam_data["width"]
            img_h = cam_data["height"]

            # Unique filename stem: scene__camera__timestamp
            stem     = f"{scene['name']}__{cam}__{cam_data['timestamp']}"
            src_img  = Path(nusc.dataroot) / cam_data["filename"]
            dest_img = images_dir / f"{stem}.jpg"
            label_path = labels_dir / f"{stem}.txt"

            label_lines = []

            for ann_token in sample["anns"]:
                ann      = nusc.get("sample_annotation", ann_token)
                class_id = get_class_id(ann["category_name"])
                if class_id is None:
                    continue

                box    = nusc.get_box(ann_token)
                result = _project_box(box, cs_record, ego_record, K)
                if result is None:
                    continue

                corners_2d, depths = result

                # Require at least half the corners to be in front of camera
                if (depths > 0).sum() < 4:
                    continue

                yolo = _corners_to_yolo(corners_2d, img_w, img_h)
                if yolo is None:
                    continue

                cx, cy, w, h = yolo
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Copy image (preserves the original; symlink is faster if disk is tight)
            if weather_pipeline is not None and split == "train" and label_lines:
                img = cv2.imread(str(src_img))
                if img is not None:
                    bboxes_yolo = []
                    class_ids   = []
                    for line in label_lines:
                        parts = line.split()
                        class_ids.append(int(parts[0]))
                        bboxes_yolo.append(tuple(float(x) for x in parts[1:]))

                    result = weather_pipeline(image=img, bboxes=bboxes_yolo, class_labels=class_ids)
                    cv2.imwrite(str(dest_img), result["image"])
                    label_lines = [
                        f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                        for cls, (cx, cy, w, h) in zip(result["class_labels"], result["bboxes"])
                    ]
                else:
                    shutil.copy2(src_img, dest_img)
            else:
                if not src_img.exists():
                    print(f"Skipping missing image: {src_img}")
                    continue
                shutil.copy2(src_img, dest_img)

            # Write labels (empty file = valid image with no detectable objects)
            label_path.write_text("\n".join(label_lines))

            if visualize and label_lines:
                _draw_boxes(src_img, label_lines, img_w, img_h,
                            vis_dir / f"{stem}.jpg")

        sample_token = sample["next"]


def _draw_boxes(img_path: Path, label_lines: list[str],
                img_w: int, img_h: int, out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    for line in label_lines:
        cls_id, cx, cy, w, h = line.split()
        cls_id = int(cls_id)
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, CLASSES[cls_id], (x1, max(y1 - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(out_path), img)


def write_dataset_yaml(output_dir: Path) -> None:
    lines = [
        f"path:  {output_dir.resolve()}",
        "train: images/train",
        "val:   images/val",
        f"nc:    {NUM_CLASSES}",
        "names:",
    ]
    for i, name in enumerate(CLASSES):
        lines.append(f"  {i}: {name}")
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {yaml_path}")


# Keywords that indicate non-clear conditions in nuScenes scene descriptions.
_BAD_WEATHER = {"rain", "rainy", "wet", "night", "dark", "fog", "foggy", "snow", "snowy"}


def is_clear_weather(scene: dict) -> bool:
    """
    Return True if the scene description contains no bad-weather keywords.
    nuScenes descriptions are free-form (e.g. "Parked car, night, rain"),
    so we tokenize and check against a blocklist.
    """
    tokens = scene.get("description", "").lower().replace(",", " ").split()
    return not any(t in _BAD_WEATHER for t in tokens)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert nuScenes 3D annotations to YOLO 2D format"
    )
    parser.add_argument("--dataroot",     required=True,
                        help="nuScenes dataset root (contains 'samples/', 'v1.0-*/', etc.)")
    parser.add_argument("--version",      default="v1.0-trainval",
                        help="nuScenes split to load (default: v1.0-trainval)")
    parser.add_argument("--output",       required=True,
                        help="Destination directory for YOLO dataset")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of scenes held out for val (default: 0.1)")
    parser.add_argument("--clear-only",   action="store_true",
                        help="Keep only clear-weather daytime scenes (filters rain/night/fog)")
    parser.add_argument("--visualize",    action="store_true",
                        help="Save annotated images to visualizations/ for a sanity check")
    parser.add_argument("--weather-aug", action="store_true",
                    help="Apply albumentations weather augmentation to training images")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading nuScenes {args.version} from {args.dataroot} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    scenes = nusc.scene

    if args.clear_only:
        before = len(scenes)
        scenes = [s for s in scenes if is_clear_weather(s)]
        print(f"Clear-weather filter: {before} → {len(scenes)} scenes kept")

    n_val        = max(1, int(len(scenes) * args.val_fraction))
    train_scenes = scenes[:-n_val]
    val_scenes   = scenes[-n_val:]
    print(f"Train scenes: {len(train_scenes)}  |  Val scenes: {len(val_scenes)}")
    
    for split, scene_list in [("train", train_scenes), ("val", val_scenes)]:
        print(f"\nProcessing {split} ...")
        pipeline = get_weather_pipeline() if (args.weather_aug and HAS_ALBUMENTATIONS and split == "train") else None
        for scene in tqdm(scene_list, unit="scene"):
            process_scene(nusc, scene, split, output_dir, args.visualize, weather_pipeline=pipeline)

    write_dataset_yaml(output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
