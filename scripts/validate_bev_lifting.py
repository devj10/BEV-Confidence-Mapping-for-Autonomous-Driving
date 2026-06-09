#!/usr/bin/env python3
"""
Validate BEV lifting pipeline on a real sample.

Pipeline:
  1. Load MC detections from JSON
  2. Get sample from nuScenes
  3. Lift detections to 3D ego frame (GT-depth or LiDAR-depth mode)
  4. Project onto BEV grid
  5. Visualize: grid with detected objects + GT boxes
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from nuscenes.nuscenes import NuScenes
from bev import (
    world_to_cell, create_empty_grid, GRID_X_MAX, GRID_Y_MAX, GRID_Y_MIN,
)
from bev.lift_to_3d import (
    lift_detection_to_3d_gt_depth,
    lift_detections_batch_lidar_depth,
    get_frame_calibration,
    get_camera_intrinsics,
    get_camera_extrinsics,
    project_gt_box_centers_to_image,
)
from bev.lidar_project import extract_depth_per_detection_devkit


def load_mc_detections(json_path: str) -> List[Dict]:
    """Load MC detections from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # data is [[{detection1}, {detection2}, ...]]
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        return data[0]
    return data


def get_gt_boxes(nusc: NuScenes, sample_token: str, ego_pose: Dict) -> List[Dict]:
    """
    Extract ground truth boxes for a sample and transform to ego frame.
    
    Args:
        nusc: NuScenes instance
        sample_token: sample token
        ego_pose: ego_pose dict with rotation and translation
    
    Returns:
        List of {"location": [x, y, z], "size": [w, h, l], "category_name": ...}
        in ego frame coordinates
    """
    from pyquaternion import Quaternion
    
    sample = nusc.get('sample', sample_token)
    gt_boxes = []
    
    ego_rotation = Quaternion(ego_pose['rotation'])
    ego_translation = np.array(ego_pose['translation'])
    
    for annotation_token in sample['anns']:
        ann = nusc.get('sample_annotation', annotation_token)
        
        location_global = np.array(ann['translation'])
        
        location_ego = ego_rotation.inverse.rotate(location_global - ego_translation)
        
        size = ann['size']
        category_name = ann['category_name']
        
        gt_boxes.append({
            'token': annotation_token,
            'location': location_ego.tolist(),
            'size': size,
            'category_name': category_name,
        })
    
    return gt_boxes


def lift_detections(
    detections: List[Dict],
    gt_boxes: List[Dict],
    calib: Dict,
    ego_pose: Dict,
    nusc=None,
    sample_token: str = None,
    depth_mode: str = "gt",
) -> List[Tuple[float, float, float, float, bool]]:
    """
    Lift detections to 3D ego frame.

    depth_mode: "gt" uses GT box depth; "lidar" uses robust LiDAR depth.

    Returns:
        List of (x_m, y_m, sigma_x, sigma_y, success)
    """
    if depth_mode == "lidar":
        K = get_camera_intrinsics(calib)
        R, t = get_camera_extrinsics(calib, ego_pose)
        depth_results = extract_depth_per_detection_devkit(
            nusc, sample_token, detections, K, R, t, method="median"
        )
        raw = lift_detections_batch_lidar_depth(detections, depth_results, calib, ego_pose)
        results = []
        for r in raw:
            if r is not None:
                results.append((r["x_m"], r["y_m"], r["sigma_x"], r["sigma_y"], True))
            else:
                results.append((0, 0, 0, 0, False))
        return results

    results = []
    for det in detections:
        result = lift_detection_to_3d_gt_depth(det, gt_boxes, calib, ego_pose)
        if result is not None:
            x, y, sx, sy = result
            results.append((x, y, sx, sy, True))
        else:
            results.append((0, 0, 0, 0, False))
    return results


def nearest_gt_errors(
    detections_3d: List[Tuple],
    gt_boxes: List[Dict],
) -> List[Optional[float]]:
    """Return nearest vehicle-GT distance (m) for each successful detection."""
    veh_gt = [g for g in gt_boxes if g["category_name"].startswith("vehicle.")]
    errors = []
    for x, y, sx, sy, ok in detections_3d:
        if not ok:
            errors.append(None)
            continue
        if not veh_gt:
            errors.append(None)
            continue
        dists = [np.hypot(x - g["location"][0], y - g["location"][1]) for g in veh_gt]
        errors.append(float(min(dists)))
    return errors


def project_onto_bev_grid(
    detections_3d: List[Tuple[float, float, float, float, bool]],
    sigma_thresh: float = float("inf"),
) -> np.ndarray:
    """
    Project 3D detections onto BEV grid.

    sigma_thresh: skip detections with sigma > this value (meters).
                  Use float('inf') to keep all. Recommended: 3.0 for LiDAR depth.
    """
    grid = create_empty_grid()

    for x, y, sx, sy, success in detections_3d:
        if not success:
            continue
        if sx > sigma_thresh:
            continue

        cell = world_to_cell(x, y)
        if cell is None:
            continue

        row, col = cell
        r_size, c_size = 2, 2
        r_min = max(0, row - r_size)
        r_max = min(grid.shape[0], row + r_size + 1)
        c_min = max(0, col - c_size)
        c_max = min(grid.shape[1], col + c_size + 1)
        grid[r_min:r_max, c_min:c_max] = 1.0

    return grid


def project_gt_onto_bev(gt_boxes: List[Dict]) -> np.ndarray:
    """
    Project GT boxes onto BEV grid.
    
    Returns:
        200×200 grid with GT boxes as rectangles
    """
    grid = create_empty_grid()
    
    for gt in gt_boxes:
        location = gt.get('location')
        size = gt.get('size')
        
        if location is None or size is None:
            continue
        
        x, y, z = location[0], location[1], location[2]
        # nuScenes size = [width, length, height]
        # width → y-axis (lateral), length → x-axis (forward/back)
        width, length, height = size[0], size[1], size[2]

        x_min, x_max = x - length / 2, x + length / 2
        y_min, y_max = y - width / 2, y + width / 2
        
        # Convert corners to grid cells
        cell_min = world_to_cell(x_min, y_min)
        cell_max = world_to_cell(x_max, y_max)
        
        if cell_min is None or cell_max is None:
            continue
        
        r_min, c_min = cell_min
        r_max, c_max = cell_max
        
        # Draw rectangle on grid (handle reversed min/max)
        r_min, r_max = sorted([r_min, r_max])
        c_min, c_max = sorted([c_min, c_max])
        
        if r_min >= 0 and r_max < 200 and c_min >= 0 and c_max < 200:
            grid[r_min:r_max+1, c_min:c_max+1] = 0.5
    
    return grid


def visualize_bev(
    det_grid: np.ndarray,
    gt_grid: np.ndarray,
    detections_3d: List[Tuple[float, float, float, float, bool]],
    output_path: str = "bev_validation.png",
):
    """
    Visualize BEV grids: GT boxes + detections with better clarity.
    
    Args:
        det_grid: Detection grid (white dots)
        gt_grid: Ground truth grid (gray boxes)
        detections_3d: List of (x, y, sx, sy, success) for uncertainty coloring
        output_path: Output PNG file
    """
    fig = plt.figure(figsize=(16, 10))
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    def _setup_bev_axes(ax, title):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('← ego left (y=+25m)   |   ego right (y=−25m) →')
        ax.set_ylabel('← ego (0 m)   |   forward (50 m) →')
        ax.set_xlim(200, 0)
        ax.set_ylim(0, 200)

        # Tick labels in meters
        tick_cells = [0, 40, 80, 120, 160, 200]
        ax.set_xticks(tick_cells)
        ax.set_xticklabels([f"{int(25 - c * 0.25)}m" for c in tick_cells])
        ax.set_yticks(tick_cells)
        ax.set_yticklabels([f"{int(c * 0.25)}m" for c in tick_cells])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(gt_grid, cmap='Blues', origin='lower', vmin=0, vmax=1, alpha=0.7)
    _setup_bev_axes(ax0, 'Ground Truth Boxes (BEV)')

    ax1 = fig.add_subplot(gs[0, 1])

    det_colored = np.zeros((200, 200, 3), dtype=np.float32)
    SIGMA_RELIABLE = 3.0  # below this = green, above = red

    from bev import world_to_cell
    for x, y, sx, sy, success in detections_3d:
        if not success:
            continue
        cell = world_to_cell(x, y)
        if cell is None:
            continue

        row, col = cell
        r_size, c_size = 1, 1
        r_min = max(0, row - r_size); r_max = min(200, row + r_size + 1)
        c_min = max(0, col - c_size); c_max = min(200, col + c_size + 1)

        if sx <= SIGMA_RELIABLE:
            det_colored[r_min:r_max, c_min:c_max, 1] = 1.0   # green = reliable
        else:
            det_colored[r_min:r_max, c_min:c_max, 0] = 1.0   # red = noisy/FP

    ax1.imshow(det_colored, origin='lower')
    _setup_bev_axes(ax1, f'Lifted Detections (green σ≤{SIGMA_RELIABLE}m, red=noisy)')

    # ===== Panel 3: Overlay GT (blue) + Detections (green) =====
    ax2 = fig.add_subplot(gs[1, :])

    overlay = np.zeros((200, 200, 3), dtype=np.float32)
    overlay[:, :, 2] = gt_grid * 0.6        # GT in blue
    overlay[:, :, 1] = det_colored[:, :, 1]  # Detections in green

    ax2.imshow(overlay, origin='lower')

    for i in range(0, 201, 40):  # Every 40 cells = 10 m
        ax2.axhline(y=i, color='white', linestyle='--', alpha=0.2, linewidth=0.5)
        ax2.axvline(x=i, color='white', linestyle='--', alpha=0.2, linewidth=0.5)

    _setup_bev_axes(ax2, 'BEV Overlay: GT (Blue) + Detections (Green)')
    
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Saved BEV visualization to {output_path}")
    plt.close()


def visualize_gt_center_projection(
    nusc: NuScenes,
    cam_token: str,
    gt_boxes: List[Dict],
    calib: Dict,
    ego_pose: Dict,
    output_path: str,
):
    """Validation gate #1: draw projected GT centers on the camera image."""
    image_path = nusc.get_sample_data_path(cam_token)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(image_path)

    image_h, image_w = image_bgr.shape[:2]
    projections = project_gt_box_centers_to_image(
        gt_boxes,
        calib,
        ego_pose,
        image_shape=(image_h, image_w),
        category_prefixes=("vehicle.",),
    )

    for projection in projections:
        u, v = projection["uv"]
        depth = projection["depth"]
        cv2.circle(image_bgr, (int(round(u)), int(round(v))), 5, (0, 255, 0), -1)
        cv2.putText(
            image_bgr,
            f"{depth:.1f}m",
            (int(round(u)) + 6, int(round(v)) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, image_bgr)
    print(f"✓ Saved GT projection validation to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate BEV lifting on real sample")
    parser.add_argument("--sample-token", default=None)
    parser.add_argument("--detections-json", default="results/mc_finetuned/mc_detections.json")
    parser.add_argument("--output-dir", default="results/bev_validation")
    parser.add_argument(
        "--depth-mode", default="lidar", choices=["gt", "lidar"],
        help="Depth source: 'gt' uses GT box depth, 'lidar' uses LiDAR point cloud",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"BEV LIFTING VALIDATION  [depth-mode={args.depth_mode}]")
    print("=" * 80)

    print("\n[1] Loading nuScenes dataset...")
    nusc = NuScenes(version='v1.0-mini', dataroot='data/v1.0-mini', verbose=False)
    print("✓ Dataset loaded")

    if args.sample_token is None:
        sample_token = nusc.sample[0]['token']
        print(f"\nAuto-selected sample: {sample_token[:8]}...")
    else:
        sample_token = args.sample_token

    sample = nusc.get('sample', sample_token)
    print(f"✓ Sample: {sample['timestamp']}")

    print("\n[2] Loading camera calibration...")
    cam_token = sample['data']['CAM_FRONT']
    frame_calib = get_frame_calibration(nusc, sample_token, "CAM_FRONT")
    cam_data = frame_calib["sample_data"]
    calib = frame_calib["calibrated_sensor"]
    ego_pose = frame_calib["ego_pose"]
    print("✓ Camera calibration loaded")

    print("\n[3] Loading ground truth boxes...")
    gt_boxes = get_gt_boxes(nusc, sample_token, ego_pose)
    print(f"✓ Found {len(gt_boxes)} GT boxes")

    gate1_path = output_dir / "gt_center_projection.png"
    visualize_gt_center_projection(nusc, cam_token, gt_boxes, calib, ego_pose, str(gate1_path))

    print("\n[4] Loading MC detections...")
    if not Path(args.detections_json).exists():
        print(f"⚠ Detections file not found: {args.detections_json}")
        sys.exit(1)

    detections = load_mc_detections(args.detections_json)
    print(f"✓ Loaded {len(detections)} MC detections")

    print(f"\n[5] Lifting detections to 3D ego frame  ({args.depth_mode} depth)...")
    detections_3d = lift_detections(
        detections, gt_boxes, calib, ego_pose,
        nusc=nusc, sample_token=sample_token, depth_mode=args.depth_mode,
    )

    successful = sum(1 for *_, ok in detections_3d if ok)
    print(f"✓ Successfully lifted {successful}/{len(detections_3d)} detections")

    # Per-detection error table
    errors = nearest_gt_errors(detections_3d, gt_boxes)
    print(f"\n{'Idx':<4} {'X (m)':<8} {'Y (m)':<8} {'σx':<6} {'σy':<6} {'NearestGT':<12} {'OK'}")
    print("-" * 55)
    for i, ((x, y, sx, sy, ok), err) in enumerate(zip(detections_3d, errors)):
        err_s = f"{err:.2f} m" if err is not None else "  —  "
        print(f"{i:<4} {x:<8.1f} {y:<8.1f} {sx:<6.2f} {sy:<6.2f} {err_s:<12} {'✓' if ok else '✗'}")

    valid_errors = [e for e in errors if e is not None and e < 10.0]
    if valid_errors:
        print(f"\nAll:     median {np.median(valid_errors):.2f} m  mean {np.mean(valid_errors):.2f} m  (excl >10 m)")

    # σ-filtered stats (LiDAR depth quality gate)
    SIGMA_GATE = 3.0
    filtered = [(x, y, sx, sy, ok, err) for (x, y, sx, sy, ok), err
                in zip(detections_3d, errors) if ok and sx <= SIGMA_GATE]
    if filtered:
        filt_errs = [e for *_, e in filtered if e is not None and e < 10.0]
        print(f"σ≤{SIGMA_GATE}m:  {len(filtered)} dets  median {np.median(filt_errs):.2f} m  mean {np.mean(filt_errs):.2f} m" if filt_errs else "")

    # Project onto BEV grids
    print("\n[6] Projecting onto BEV grids...")
    sigma_thresh = SIGMA_GATE if args.depth_mode == "lidar" else float("inf")
    det_grid = project_onto_bev_grid(detections_3d, sigma_thresh=sigma_thresh)
    gt_grid = project_gt_onto_bev(gt_boxes)
    print(f"✓ Projected detections (σ≤{sigma_thresh:.0f}m gate): {np.sum(det_grid > 0)} cells")
    print(f"✓ Projected GT boxes: {np.sum(gt_grid > 0)} cells")

    # Visualize
    print("\n[7] Visualizing BEV...")
    viz_path = output_dir / "bev_validation.png"
    visualize_bev(det_grid, gt_grid, detections_3d, str(viz_path))
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  • bev_validation.png: BEV grid visualization")
    print(f"  • gt_center_projection.png: GT centers projected onto image")


if __name__ == "__main__":
    main()
