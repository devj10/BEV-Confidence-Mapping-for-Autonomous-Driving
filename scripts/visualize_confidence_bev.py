#!/usr/bin/env python3
"""
visualize_confidence_bev.py — BEV with confidence heatmap overlay.

Renders the clean dark BEV (GT boxes + grid) and overlays a smooth
Gaussian confidence heatmap from MC-DropBlock detections lifted to 3D
from all 6 cameras.

Usage:
    python scripts/visualize_confidence_bev.py \
        --model model_final.pt \
        --sample-token <token> \
        --output results/bev_confidence_overlay.png
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).parent.parent))

from nuscenes.nuscenes import NuScenes
from ultralytics import YOLO
from bev.bev_grid import GRID_X_MIN, GRID_X_MAX, GRID_Y_MIN, GRID_Y_MAX
from bev.lift_to_3d import (
    get_frame_calibration, get_camera_intrinsics, get_camera_extrinsics,
    lift_detections_batch_lidar_depth,
)
from bev.lidar_project import extract_depth_per_detection_devkit
from scripts.visualize_rgb_bev import (
    get_gt_boxes_ego, rotated_rect_corners, category_color,
)

ALL_CAMS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

CAM_COLORS = {
    "CAM_FRONT":       "#ffffff",
    "CAM_FRONT_LEFT":  "#aed6f1",
    "CAM_FRONT_RIGHT": "#a9dfbf",
    "CAM_BACK":        "#f1948a",
    "CAM_BACK_LEFT":   "#d7bde2",
    "CAM_BACK_RIGHT":  "#f9e79f",
}

CONFIDENCE_CMAP = LinearSegmentedColormap.from_list(
    "conf_heat",
    [
        (0.00, "#000000"),
        (0.20, "#0d0d6e"),
        (0.45, "#7b00d4"),
        (0.65, "#ff6600"),
        (0.85, "#ffcc00"),
        (1.00, "#ffffff"),
    ],
)


def load_model(model_path):
    from inject_dropblock import inject_dropblock
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to("cuda")
    injection = inject_dropblock(model.model)
    for db in injection.dropblocks:
        db.mc_inference = True
    class_names = model.names  # {0: 'car', 1: 'truck', ...}
    return model, class_names


def run_mc_inference(model, image_path, num_passes=10, conf_thresh=0.3):
    results_all = []
    for _ in range(num_passes):
        results = model(str(image_path), conf=conf_thresh, verbose=False)
        dets = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    dets.append({
                        "xyxy": box.xyxy[0].tolist(),
                        "conf": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                    })
        results_all.append(dets)
    # Average across passes: flatten and return unique boxes by NMS-like dedup
    all_dets = [d for pass_dets in results_all for d in pass_dets]
    return all_dets


def lift_to_bev(nusc, sample_token, detections, calib, ego_pose, class_names=None):
    if not detections:
        return []
    K = get_camera_intrinsics(calib)
    R, t = get_camera_extrinsics(calib, ego_pose)
    depth_results = extract_depth_per_detection_devkit(
        nusc, sample_token, detections, K, R, t
    )
    raw = lift_detections_batch_lidar_depth(detections, depth_results, calib, ego_pose)
    lifted = []
    for i, r in enumerate(raw):
        if r is not None:
            conf     = detections[i].get("conf", 1.0)
            cls_id   = detections[i].get("class_id", -1)
            cls_name = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)
            lifted.append((r["x_m"], r["y_m"], r["sigma_x"], conf, cls_name))
    return lifted


def build_heatmap(lifted, grid_shape, x_range, y_range, sigma_m=2.5):
    H, W = grid_shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    x_min, x_max = x_range
    y_min, y_max = y_range
    for x_m, y_m, sigma, conf, *_ in lifted:
        if not (x_min <= x_m <= x_max and y_min <= y_m <= y_max):
            continue
        col = int((y_m - y_min) / (y_max - y_min) * W)
        row = int((x_max - x_m) / (x_max - x_min) * H)
        col = np.clip(col, 0, W - 1)
        row = np.clip(row, 0, H - 1)
        heatmap[row, col] += conf
    px_per_m = H / (x_max - x_min)
    heatmap = gaussian_filter(heatmap, sigma=sigma_m * px_per_m)
    return heatmap


def fov_cone_for_camera(calib, fov_deg=70, max_range=50):
    """Draw a FOV wedge for a camera using its actual ego-frame orientation."""
    rot = Quaternion(calib["rotation"])
    # nuScenes camera optical axis is +z in sensor frame; rotate to ego frame
    cam_forward_ego = rot.rotate(np.array([0.0, 0.0, 1.0]))
    # ego: x=forward, y=left. BEV plot: plot_x = -ego_y (flipped), plot_y = ego_x
    heading_plot = np.arctan2(cam_forward_ego[0], -cam_forward_ego[1])  # angle in plot space

    half = np.radians(fov_deg / 2)
    angles = np.linspace(heading_plot - half, heading_plot + half, 40)
    px = np.concatenate([[0], max_range * np.cos(angles), [0]])
    py = np.concatenate([[0], max_range * np.sin(angles), [0]])
    # Offset by camera translation — negate lateral (ego y → -plot_x since left=left)
    tx, ty = -calib["translation"][1], calib["translation"][0]
    return px + tx, py + ty


def render(nusc, sample_token, model_path, output_path,
           scene_name="", scene_desc="", timestamp="", num_passes=10):

    sample  = nusc.get("sample", sample_token)
    # Use CAM_FRONT ego pose as the reference frame for all cameras
    ref_calib = get_frame_calibration(nusc, sample_token, "CAM_FRONT")
    ego_pose  = ref_calib["ego_pose"]
    gt_boxes  = get_gt_boxes_ego(nusc, sample_token, ego_pose)

    print("Loading model...")
    model, class_names = load_model(model_path)

    all_lifted = []
    cam_lifted_counts = {}

    for cam in ALL_CAMS:
        cam_data = nusc.get("sample_data", sample["data"][cam])
        image_path = Path("data/v1.0-mini") / cam_data["filename"]
        calib_rec  = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

        print(f"  [{cam}] running {num_passes} MC passes on {image_path.name}...")
        detections = run_mc_inference(model, image_path, num_passes=num_passes)
        lifted     = lift_to_bev(nusc, sample_token, detections, calib_rec, ego_pose,
                                 class_names=class_names)
        all_lifted.extend(lifted)
        cam_lifted_counts[cam] = len(lifted)
        print(f"    → {len(detections)} raw dets, {len(lifted)} lifted to BEV")

    print(f"\nTotal lifted detections across all cameras: {len(all_lifted)}")

    x_range = (GRID_X_MIN, GRID_X_MAX)
    y_range = (GRID_Y_MIN, GRID_Y_MAX)
    H, W = 400, 400
    heatmap = build_heatmap(all_lifted, (H, W), x_range, y_range, sigma_m=2.5)
    hmax = heatmap.max()
    if hmax > 0:
        heatmap /= hmax

    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 12), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    # Heatmap — extent uses flipped lateral axis (-GRID_Y_MAX to -GRID_Y_MIN)
    im = ax.imshow(
        np.fliplr(heatmap),
        extent=[GRID_Y_MIN, GRID_Y_MAX, GRID_X_MIN, GRID_X_MAX],
        origin="upper",
        cmap=CONFIDENCE_CMAP,
        vmin=0, vmax=1,
        alpha=0.75,
        interpolation="bilinear",
        zorder=1,
    )

    # FOV cones for all 6 cameras
    for cam in ALL_CAMS:
        cam_data  = nusc.get("sample_data", sample["data"][cam])
        calib_rec = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        color = CAM_COLORS[cam]
        px, py = fov_cone_for_camera(calib_rec, fov_deg=70, max_range=50)
        ax.fill(px, py, color=color, alpha=0.04, zorder=2)
        ax.plot(px, py, color=color, alpha=0.25, linewidth=0.8, zorder=2)

    # Grid
    for v in range(int(GRID_Y_MIN), int(GRID_Y_MAX) + 1, 10):
        ax.axvline(v, color="#ffffff", alpha=0.07, linewidth=0.4, zorder=0)
    for v in range(int(GRID_X_MIN), int(GRID_X_MAX) + 1, 10):
        ax.axhline(v, color="#ffffff", alpha=0.07, linewidth=0.4, zorder=0)
    for r in [15, 30, 50]:
        ring = plt.Circle((0, 0), r, color="#ffffff", fill=False,
                           linewidth=0.5, alpha=0.15, linestyle="--", zorder=0)
        ax.add_patch(ring)
        ax.text(0.5, r + 0.5, f"{r} m", color="#ffffff", alpha=0.35,
                fontsize=7, ha="center", zorder=0)

    # GT boxes — negate y to flip lateral axis (left of vehicle → left of plot)
    for gt in gt_boxes:
        loc, size, yaw, cat = gt["location"], gt["size"], gt["yaw"], gt["category_name"]
        x, y = loc[0], loc[1]
        if not (GRID_X_MIN <= x <= GRID_X_MAX and GRID_Y_MIN <= y <= GRID_Y_MAX):
            continue
        length, width = size[1], size[0]
        color = category_color(cat)
        lw = 1.6 if cat.startswith("vehicle.") else 0.9
        xs, ys = rotated_rect_corners(-y, x, width, length, -yaw)
        ax.add_patch(plt.Polygon(
            np.column_stack([xs, ys]), closed=True,
            edgecolor=color, facecolor="none", linewidth=lw, alpha=0.9, zorder=4,
        ))

    # Detection dots + class labels — negate y_m to match flipped axis
    label_positions = []
    for x_m, y_m, sigma, conf, cls_name in all_lifted:
        if not (GRID_X_MIN <= x_m <= GRID_X_MAX and GRID_Y_MIN <= y_m <= GRID_Y_MAX):
            continue
        plot_x = -y_m
        ax.scatter(plot_x, x_m, c=[[*CONFIDENCE_CMAP(conf)]], s=60, zorder=6,
                   edgecolors="white", linewidths=0.5)

        too_close = any(abs(plot_x - lx) < 3 and abs(x_m - ly) < 3
                        for lx, ly in label_positions)
        if not too_close:
            ax.annotate(
                cls_name,
                xy=(plot_x, x_m),
                xytext=(plot_x + 1.2, x_m + 1.2),
                color="white",
                fontsize=6.5,
                fontweight="bold",
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#000000",
                          edgecolor="none", alpha=0.55),
            )
            label_positions.append((plot_x, x_m))

    # Ego vehicle
    ax.fill([-1, 1, 1, -1, -1], [-1.2, -1.2, 2.5, 2.5, -1.2],
            color="#ffd600", alpha=0.95, zorder=7)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Detection Confidence", color="#cccccc", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#4fc3f7", edgecolor="#4fc3f7", alpha=0.7, label="GT vehicle"),
        mpatches.Patch(facecolor="#ff8a65", edgecolor="#ff8a65", alpha=0.7, label="GT pedestrian"),
        mpatches.Patch(facecolor="#ffd600", label="Ego vehicle"),
        plt.Line2D([0], [0], marker="o", color="w", markersize=7,
                   markeredgecolor="white", label=f"{len(all_lifted)} lifted dets (all cams)"),
    ]
    # Camera FOV legend entries
    for cam, color in CAM_COLORS.items():
        n = cam_lifted_counts.get(cam, 0)
        legend_elements.append(
            plt.Line2D([0], [0], color=color, linewidth=1.5, alpha=0.8,
                       label=f"{cam.replace('CAM_', '')} ({n} dets)")
        )
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor="#111122", labelcolor="#dddddd", fontsize=7, framealpha=0.85)

    ax.set_xlim(GRID_Y_MIN, GRID_Y_MAX)
    ax.set_ylim(GRID_X_MIN, GRID_X_MAX)
    ax.set_xlabel("Lateral  ← left  |  right →  (m)", color="#cccccc", fontsize=10)
    ax.set_ylabel("Forward (m)", color="#cccccc", fontsize=10)
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_title(
        f"Bird's-Eye View — All-Camera Confidence Heatmap\n"
        f"{scene_name}  |  {scene_desc}\n"
        f"ts={timestamp}",
        color="#eeeeee", fontsize=10, pad=10,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_final.pt")
    parser.add_argument("--sample-token", default=None)
    parser.add_argument("--num-passes", type=int, default=10)
    parser.add_argument("--output", default="results/bev_confidence_overlay_allcams.png")
    args = parser.parse_args()

    nusc  = NuScenes(version="v1.0-mini", dataroot="data/v1.0-mini", verbose=False)
    token = args.sample_token or nusc.sample[0]["token"]

    sample = nusc.get("sample", token)
    scene  = nusc.get("scene", sample["scene_token"])
    print(f"Scene:     {scene['name']}")
    print(f"Desc:      {scene['description']}")
    print(f"Timestamp: {sample['timestamp']}")

    render(nusc, token, args.model, args.output,
           scene_name=scene["name"], scene_desc=scene["description"],
           timestamp=sample["timestamp"], num_passes=args.num_passes)


if __name__ == "__main__":
    main()
