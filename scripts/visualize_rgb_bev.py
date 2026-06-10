#!/usr/bin/env python3
"""
Clean BEV visualization — no LiDAR point cloud.

Shows:
  - Light grey background with grid
  - Camera FOV cone (CAM_FRONT)
  - GT boxes drawn as properly-rotated rectangles, coloured by category
  - Lifted detections as circles coloured by σ (blue=reliable, red=noisy)
  - Uncertainty ring scaled to σ
  - Ego vehicle footprint
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).parent.parent))

from nuscenes.nuscenes import NuScenes
from bev.bev_grid import GRID_X_MIN, GRID_X_MAX, GRID_Y_MIN, GRID_Y_MAX
from bev.lift_to_3d import (
    get_frame_calibration, get_camera_intrinsics, get_camera_extrinsics,
    lift_detection_to_3d_gt_depth, lift_detections_batch_lidar_depth,
)
from bev.lidar_project import extract_depth_per_detection_devkit


def get_gt_boxes_ego(nusc, sample_token, ego_pose):
    sample = nusc.get("sample", sample_token)
    ego_rot = Quaternion(ego_pose["rotation"])
    ego_t = np.array(ego_pose["translation"])
    boxes = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        loc_ego = ego_rot.inverse.rotate(np.array(ann["translation"]) - ego_t)
        # Heading in ego frame: global yaw rotated by ego inverse
        ann_rot = Quaternion(ann["rotation"])
        rot_ego = ego_rot.inverse * ann_rot
        yaw = rot_ego.yaw_pitch_roll[0]  # radians, yaw = rotation about z
        boxes.append({
            "location": loc_ego.tolist(),
            "size": ann["size"],
            "yaw": yaw,
            "category_name": ann["category_name"],
        })
    return boxes


def load_detections(json_path):
    with open(json_path) as f:
        data = json.load(f)
    # run_mc_inference.py format: [{"image": ..., "detections": [...]}]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "detections" in data[0]:
        return data[0]["detections"]
    # legacy format: [[det, det, ...]]
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data[0]
    return data


def lift_dets(nusc, sample_token, detections, calib, ego_pose, depth_mode):
    if depth_mode == "lidar":
        K = get_camera_intrinsics(calib)
        R, t = get_camera_extrinsics(calib, ego_pose)
        depth_results = extract_depth_per_detection_devkit(
            nusc, sample_token, detections, K, R, t
        )
        raw = lift_detections_batch_lidar_depth(detections, depth_results, calib, ego_pose)
        out = []
        for r in raw:
            if r is not None:
                out.append((r["x_m"], r["y_m"], r["sigma_x"], True))
            else:
                out.append((0, 0, 0, False))
        return out

    gt_boxes = get_gt_boxes_ego(nusc, sample_token, ego_pose)
    out = []
    for det in detections:
        r = lift_detection_to_3d_gt_depth(det, gt_boxes, calib, ego_pose)
        if r:
            out.append((r[0], r[1], r[2], True))
        else:
            out.append((0, 0, 0, False))
    return out


def rotated_rect_corners(cx, cy, length, width, yaw):
    """Return 4 corners of a rotated rectangle. yaw in radians."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    dx = np.array([ length/2,  length/2, -length/2, -length/2])
    dy = np.array([ width/2,  -width/2,  -width/2,   width/2])
    xs = cx + cos_y * dx - sin_y * dy
    ys = cy + sin_y * dx + cos_y * dy
    return xs, ys


def cam_front_fov_patch(fov_deg=70, max_range=50):
    """Return a wedge patch for the CAM_FRONT field of view."""
    half = np.radians(fov_deg / 2)
    # Camera points forward (ego +x), so the wedge is centred on 90° in plot coords
    # In BEV: x=forward (plot y-axis), y=lateral (plot x-axis)
    angles = np.linspace(np.pi/2 - half, np.pi/2 + half, 60)
    xs = np.concatenate([[0], max_range * np.cos(angles), [0]])
    ys = np.concatenate([[0], max_range * np.sin(angles), [0]])
    # Convert to plot coords: plot_x = ego_y, plot_y = ego_x
    return ys, xs  # (plot_x_array, plot_y_array)


CATEGORY_COLORS = {
    "vehicle.car":          "#4fc3f7",
    "vehicle.truck":        "#29b6f6",
    "vehicle.bus":          "#0288d1",
    "vehicle.motorcycle":   "#81d4fa",
    "vehicle.bicycle":      "#b3e5fc",
    "human.pedestrian":     "#ff8a65",
    "movable_object.barrier": "#bdbdbd",
    "movable_object.trafficcone": "#ffcc02",
}
DEFAULT_VEHICLE_COLOR = "#4fc3f7"
DEFAULT_OTHER_COLOR   = "#9e9e9e"


def category_color(cat):
    for prefix, color in CATEGORY_COLORS.items():
        if cat.startswith(prefix):
            return color
    return DEFAULT_OTHER_COLOR


def render_clean_bev(
    nusc,
    sample_token,
    detections=None,
    depth_mode="lidar",
    output_path="results/bev_clean/bev.png",
    sigma_thresh=3.0,
):
    sample = nusc.get("sample", sample_token)
    frame_calib = get_frame_calibration(nusc, sample_token, "CAM_FRONT")
    calib = frame_calib["calibrated_sensor"]
    ego_pose = frame_calib["ego_pose"]

    gt_boxes = get_gt_boxes_ego(nusc, sample_token, ego_pose)

    lifted = []
    if detections:
        print("Lifting detections...")
        lifted = lift_dets(nusc, sample_token, detections, calib, ego_pose, depth_mode)

    fig, ax = plt.subplots(figsize=(10, 12), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    fov_px, fov_py = cam_front_fov_patch(fov_deg=70, max_range=50)
    ax.fill(fov_px, fov_py, color="#ffffff", alpha=0.04, zorder=1)
    ax.plot(fov_px, fov_py, color="#ffffff", alpha=0.15, linewidth=0.8, zorder=1)

    for v in range(int(GRID_Y_MIN), int(GRID_Y_MAX) + 1, 10):
        ax.axvline(v, color="#ffffff", alpha=0.08, linewidth=0.5, zorder=0)
    for v in range(int(GRID_X_MIN), int(GRID_X_MAX) + 1, 10):
        ax.axhline(v, color="#ffffff", alpha=0.08, linewidth=0.5, zorder=0)
    # Range rings
    for r in [15, 30, 50]:
        ring = plt.Circle((0, 0), r, color="#ffffff", fill=False,
                           linewidth=0.5, alpha=0.12, linestyle="--", zorder=0)
        ax.add_patch(ring)
        ax.text(0.5, r + 0.5, f"{r} m", color="#ffffff", alpha=0.3,
                fontsize=7, ha="center", zorder=0)

    for gt in gt_boxes:
        loc = gt["location"]
        size = gt["size"]
        yaw = gt["yaw"]
        cat = gt["category_name"]

        x, y = loc[0], loc[1]
        if not (GRID_X_MIN <= x <= GRID_X_MAX and GRID_Y_MIN <= y <= GRID_Y_MAX):
            continue

        length = size[1]  # forward/back
        width  = size[0]  # lateral

        color = category_color(cat)
        is_vehicle = cat.startswith("vehicle.")
        lw = 1.4 if is_vehicle else 0.8
        alpha = 0.9 if is_vehicle else 0.5

        xs, ys = rotated_rect_corners(y, x, width, length, yaw)
        poly = plt.Polygon(
            np.column_stack([xs, ys]),
            closed=True, edgecolor=color, facecolor=color,
            alpha=alpha * 0.25, linewidth=lw, zorder=2,
        )
        ax.add_patch(poly)
        ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]),
                color=color, linewidth=lw, alpha=alpha, zorder=3)

        arrow_len = length * 0.45
        ax.annotate(
            "", xy=(y + np.sin(yaw) * arrow_len, x + np.cos(yaw) * arrow_len),
            xytext=(y, x),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, alpha=alpha),
            zorder=3,
        )

    for x_m, y_m, sigma, ok in lifted:
        if not ok:
            continue
        reliable = sigma <= sigma_thresh
        if reliable:
            color = "#00e5ff"
            ring_r = max(sigma, 0.5)
            ax.scatter(y_m, x_m, c=color, s=120, marker="o",
                       edgecolors="white", linewidths=0.8, zorder=6)
            ring = plt.Circle((y_m, x_m), ring_r, color=color,
                               fill=False, linewidth=1.2, alpha=0.6, zorder=5)
            ax.add_patch(ring)
        else:
            ax.scatter(y_m, x_m, c="#ff5252", s=70, marker="o",
                       edgecolors="#ff5252", linewidths=0.5, alpha=0.6, zorder=6)

    ego_box_xs = [-1, 1, 1, -1, -1]
    ego_box_ys = [-1.2, -1.2, 2.5, 2.5, -1.2]
    ax.fill(ego_box_xs, ego_box_ys, color="#ffd600", alpha=0.9, zorder=7)
    ax.plot(ego_box_xs, ego_box_ys, color="#ffd600", linewidth=1.5, zorder=7)

    ax.set_xlim(GRID_Y_MIN, GRID_Y_MAX)
    ax.set_ylim(GRID_X_MIN, GRID_X_MAX)
    ax.set_xlabel("Lateral  ← left  |  right →  (m)", color="#cccccc", fontsize=10)
    ax.set_ylabel("Forward (m)", color="#cccccc", fontsize=10)
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    legend_elements = [
        mpatches.Patch(facecolor="#4fc3f7", edgecolor="#4fc3f7", alpha=0.7, label="GT vehicle"),
        mpatches.Patch(facecolor="#ff8a65", edgecolor="#ff8a65", alpha=0.7, label="GT pedestrian"),
        mpatches.Patch(facecolor="#9e9e9e", edgecolor="#9e9e9e", alpha=0.5, label="GT other"),
        plt.Line2D([0], [0], marker="o", color="#00e5ff", markersize=8,
                   markeredgecolor="white", label=f"Det reliable (σ≤{sigma_thresh}m)"),
        plt.Line2D([0], [0], marker="o", color="#ff5252", markersize=7,
                   linestyle="none", label="Det noisy"),
        mpatches.Patch(facecolor="#ffd600", label="Ego vehicle"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper right",
                    facecolor="#111122", labelcolor="#dddddd",
                    fontsize=8, framealpha=0.85)

    mode_tag = f"depth={depth_mode}" if detections else "GT only"
    ax.set_title(
        f"Bird's-Eye View  [{mode_tag}]  ts={sample['timestamp']}",
        color="#eeeeee", fontsize=11, pad=10,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"✓ Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-token", default=None)
    parser.add_argument("--detections-json", default=None)
    parser.add_argument("--depth-mode", default="lidar", choices=["gt", "lidar"])
    parser.add_argument("--output", default="results/bev_clean/bev.png")
    parser.add_argument("--sigma-thresh", type=float, default=3.0)
    args = parser.parse_args()

    nusc = NuScenes(version="v1.0-mini", dataroot="data/v1.0-mini", verbose=False)
    token = args.sample_token or nusc.sample[1]["token"]
    print(f"Sample: {token[:16]}...")

    detections = None
    if args.detections_json and Path(args.detections_json).exists():
        detections = load_detections(args.detections_json)
        print(f"Loaded {len(detections)} detections")

    render_clean_bev(
        nusc, token, detections,
        depth_mode=args.depth_mode,
        output_path=args.output,
        sigma_thresh=args.sigma_thresh,
    )


if __name__ == "__main__":
    main()
