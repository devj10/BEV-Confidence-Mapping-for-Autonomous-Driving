#!/usr/bin/env python3
"""
visualize_confidence_bev.py — Real-photo BEV via IPM + confidence heatmap.

All 6 camera images are projected onto the BEV ground plane using Inverse
Perspective Mapping (IPM), producing a photo-realistic bird's-eye view.
The MC-DropBlock confidence heatmap is overlaid on top.

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
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from nuscenes.nuscenes import NuScenes
from ultralytics import YOLO
from bev.lift_to_3d import (
    get_frame_calibration, get_camera_intrinsics, get_camera_extrinsics,
    lift_detections_batch_lidar_depth,
)
from bev.lidar_project import extract_depth_per_detection_devkit
from scripts.visualize_rgb_bev import (
    get_gt_boxes_ego, rotated_rect_corners, category_color,
)

from bev.bev_grid import GRID_X_MIN, GRID_X_MAX, GRID_Y_MIN, GRID_Y_MAX

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
    "CAM_FRONT_LEFT":  "#56ccf2",
    "CAM_FRONT_RIGHT": "#6fcf97",
    "CAM_BACK":        "#eb5757",
    "CAM_BACK_LEFT":   "#bb87fc",
    "CAM_BACK_RIGHT":  "#f2994a",
}

CAM_LABELS = {
    "CAM_FRONT":       "FRONT",
    "CAM_FRONT_LEFT":  "FRONT-L",
    "CAM_FRONT_RIGHT": "FRONT-R",
    "CAM_BACK":        "BACK",
    "CAM_BACK_LEFT":   "BACK-L",
    "CAM_BACK_RIGHT":  "BACK-R",
}

CONFIDENCE_CMAP = LinearSegmentedColormap.from_list(
    "conf_heat",
    [
        (0.00, "#0d0d6e"),
        (0.35, "#7b00d4"),
        (0.65, "#ff6600"),
        (0.85, "#ffcc00"),
        (1.00, "#ffffff"),
    ],
)


# ---------------------------------------------------------------------------
# LiDAR-guided BEV colorization
# ---------------------------------------------------------------------------

def splat_lidar_to_bev(nusc, sample_token, sample, H=520, W=400):
    """
    For each LiDAR point in 3D space:
      1. Find its true ego-frame (x, y, z) — no ground-plane assumption.
      2. Project it onto whichever camera sees it closest to center.
      3. Sample that camera's image for the pixel color.
      4. Paint the BEV cell at (x, y) with that color.

    Gaps between sparse LiDAR scan lines are filled with a gaussian splat
    (sigma ≈ 0.4 m) so the result looks continuous.

    Returns:
        bev_rgb   (H, W, 3)  float32 in [0, 1]
        coverage  (H, W)     bool
    """
    from bev.lidar_project import load_lidar_points_ego

    pts_ego = load_lidar_points_ego(nusc, sample_token)  # (N, 3)

    # Keep points in BEV range + reasonable height (-3 m to +5 m)
    in_range = (
        (pts_ego[:, 0] >= GRID_X_MIN - 1) & (pts_ego[:, 0] <= GRID_X_MAX + 1) &
        (pts_ego[:, 1] >= GRID_Y_MIN - 1) & (pts_ego[:, 1] <= GRID_Y_MAX + 1) &
        (pts_ego[:, 2] >= -3.0)            & (pts_ego[:, 2] <= 5.0)
    )
    pts = pts_ego[in_range]
    N   = len(pts)
    print(f"  LiDAR: {N} points in BEV range")

    point_colors    = np.zeros((N, 3),  dtype=np.float32)
    point_cam_score = np.full(N, -np.inf, dtype=np.float32)  # higher = preferred

    for cam in ALL_CAMS:
        cam_data  = nusc.get("sample_data", sample["data"][cam])
        img_path  = Path("data/v1.0-mini") / cam_data["filename"]
        calib_rec = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

        img = np.array(Image.open(img_path)).astype(np.float32) / 255.0
        H_img, W_img = img.shape[:2]

        K     = np.array(calib_rec["camera_intrinsic"])
        R_s2e = Quaternion(calib_rec["rotation"]).rotation_matrix  # sensor→ego
        t_s2e = np.array(calib_rec["translation"])

        # ego → camera frame:  p_cam = R_s2e.T @ (p_ego - t_s2e)
        diff  = pts - t_s2e          # (N, 3)
        p_cam = diff @ R_s2e         # (N, 3)  equivalent to R.T @ diff.T

        depth = p_cam[:, 2]
        valid = depth > 0.1

        denom = np.where(valid, depth, 1.0)
        u = K[0, 0] * p_cam[:, 0] / denom + K[0, 2]
        v = K[1, 1] * p_cam[:, 1] / denom + K[1, 2]

        valid &= (u >= 0) & (u < W_img - 1) & (v >= 0) & (v < H_img - 1)

        # Score: prefer point near image center and not too far
        cx_d  = np.abs(u - W_img / 2) / (W_img / 2)
        cy_d  = np.abs(v - H_img / 2) / (H_img / 2)
        score = np.where(valid, (1 - cx_d) * (1 - cy_d) / (depth + 1), -np.inf)

        better = score > point_cam_score

        u_b = np.clip(u[better].astype(int), 0, W_img - 2)
        v_b = np.clip(v[better].astype(int), 0, H_img - 2)
        fu  = u[better] - u_b
        fv  = v[better] - v_b

        # Bilinear interpolation
        colors = (
            img[v_b,     u_b    ] * ((1 - fu) * (1 - fv))[:, None]
          + img[v_b,     u_b + 1] * (fu        * (1 - fv))[:, None]
          + img[v_b + 1, u_b    ] * ((1 - fu) * fv        )[:, None]
          + img[v_b + 1, u_b + 1] * (fu        * fv        )[:, None]
        )

        point_colors[better]    = colors
        point_cam_score[better] = score[better]

    # --- Splat colored points into BEV grid ---
    colored_pts    = pts[point_cam_score > -np.inf]
    colored_colors = point_colors[point_cam_score > -np.inf]
    print(f"  {len(colored_pts)} points got camera colors")

    # BEV pixel indices
    rows = np.clip(
        ((GRID_X_MAX - colored_pts[:, 0]) / (GRID_X_MAX - GRID_X_MIN) * H).astype(int),
        0, H - 1,
    )
    cols = np.clip(
        ((colored_pts[:, 1] - GRID_Y_MIN) / (GRID_Y_MAX - GRID_Y_MIN) * W).astype(int),
        0, W - 1,
    )

    # Accumulate using bincount (fast, vectorized)
    flat = rows * W + cols
    bev_sum = np.zeros((H, W, 3), dtype=np.float32)
    bev_cnt = np.zeros((H, W),    dtype=np.float32)
    for ch in range(3):
        bev_sum[:, :, ch] = np.bincount(
            flat, weights=colored_colors[:, ch], minlength=H * W
        ).reshape(H, W)
    bev_cnt[:] = np.bincount(flat, minlength=H * W).reshape(H, W)

    # Gaussian splat to fill inter-scan-line gaps (~4 px ≈ 0.5 m)
    sigma_px = 3.5
    for ch in range(3):
        bev_sum[:, :, ch] = gaussian_filter(bev_sum[:, :, ch], sigma=sigma_px)
    bev_cnt_smooth = gaussian_filter(bev_cnt, sigma=sigma_px)

    covered = bev_cnt_smooth > 0.05
    bev_rgb = np.zeros((H, W, 3), dtype=np.float32)
    bev_rgb[covered] = (bev_sum[covered] / bev_cnt_smooth[covered, None])

    return np.clip(bev_rgb, 0, 1), covered


# ---------------------------------------------------------------------------
# Model + MC inference
# ---------------------------------------------------------------------------

def load_model(model_path):
    from inject_dropblock import inject_dropblock
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to("cuda")
    inj = inject_dropblock(model.model)
    for db in inj.dropblocks:
        db.mc_inference = True
    return model, model.names


def run_mc_inference(model, image_path, num_passes=10, conf_thresh=0.3):
    all_dets = []
    for _ in range(num_passes):
        for r in model(str(image_path), conf=conf_thresh, verbose=False):
            if r.boxes is not None:
                for box in r.boxes:
                    all_dets.append({
                        "xyxy":     box.xyxy[0].tolist(),
                        "conf":     float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                    })
    return all_dets


def lift_to_bev(nusc, sample_token, detections, calib, ego_pose, class_names=None):
    if not detections:
        return []
    K  = get_camera_intrinsics(calib)
    R, t = get_camera_extrinsics(calib, ego_pose)
    depth_results = extract_depth_per_detection_devkit(
        nusc, sample_token, detections, K, R, t
    )
    raw = lift_detections_batch_lidar_depth(detections, depth_results, calib, ego_pose)
    out = []
    for i, r in enumerate(raw):
        if r is not None:
            conf     = detections[i].get("conf", 1.0)
            cls_id   = detections[i].get("class_id", -1)
            cls_name = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)
            out.append((r["x_m"], r["y_m"], r["sigma_x"], conf, cls_name))
    return out


def build_heatmap(lifted, grid_shape, x_range, y_range, sigma_m=2.5):
    H, W   = grid_shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    x_min, x_max = x_range
    y_min, y_max = y_range
    for x_m, y_m, _sigma, conf, *_ in lifted:
        if not (x_min <= x_m <= x_max and y_min <= y_m <= y_max):
            continue
        col = int((y_m - y_min) / (y_max - y_min) * W)
        row = int((x_max - x_m) / (x_max - x_min) * H)
        heatmap[np.clip(row, 0, H-1), np.clip(col, 0, W-1)] += conf
    px_per_m = H / (x_max - x_min)
    return gaussian_filter(heatmap, sigma=sigma_m * px_per_m)


def fov_cone_for_camera(calib, fov_deg=70, max_range=50):
    rot      = Quaternion(calib["rotation"])
    cam_fwd  = rot.rotate(np.array([0.0, 0.0, 1.0]))
    heading  = np.arctan2(cam_fwd[0], -cam_fwd[1])
    half     = np.radians(fov_deg / 2)
    angles   = np.linspace(heading - half, heading + half, 40)
    px = np.concatenate([[0], max_range * np.cos(angles), [0]])
    py = np.concatenate([[0], max_range * np.sin(angles), [0]])
    tx, ty = -calib["translation"][1], calib["translation"][0]
    return px + tx, py + ty, heading


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render(nusc, sample_token, model_path, output_path,
           scene_name="", num_passes=10):

    sample   = nusc.get("sample", sample_token)
    ref_cal  = get_frame_calibration(nusc, sample_token, "CAM_FRONT")
    ego_pose = ref_cal["ego_pose"]
    gt_boxes = get_gt_boxes_ego(nusc, sample_token, ego_pose)

    # --- MC inference + lift ---
    H, W = 500, 500
    print("Loading model...")
    model, class_names = load_model(model_path)
    all_lifted, cam_lifted_counts = [], {}
    for cam in ALL_CAMS:
        cam_data   = nusc.get("sample_data", sample["data"][cam])
        image_path = Path("data/v1.0-mini") / cam_data["filename"]
        calib_rec  = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        print(f"  [{cam}] {num_passes} passes...")
        dets   = run_mc_inference(model, image_path, num_passes=num_passes)
        lifted = lift_to_bev(nusc, sample_token, dets, calib_rec, ego_pose,
                             class_names=class_names)
        all_lifted.extend(lifted)
        cam_lifted_counts[cam] = len(lifted)
        print(f"    → {len(dets)} raw, {len(lifted)} lifted")
    print(f"Total: {len(all_lifted)} detections")

    # --- Heatmap ---
    heatmap = build_heatmap(all_lifted, (H, W),
                            (GRID_X_MIN, GRID_X_MAX), (GRID_Y_MIN, GRID_Y_MAX))
    hmax = heatmap.max()
    if hmax > 0:
        heatmap /= hmax

    heatmap_rgba = CONFIDENCE_CMAP(np.fliplr(heatmap))   # (H,W,4)
    heatmap_rgba[..., 3] = np.fliplr(
        np.where(heatmap > 0.02, heatmap ** 0.6 * 0.88, 0.0)
    )

    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 12), facecolor="#0b0f18")
    ax.set_facecolor("#0b0f18")

    # 1. Confidence heatmap on black background
    sm = plt.cm.ScalarMappable(cmap=CONFIDENCE_CMAP,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    ax.imshow(heatmap_rgba,
              extent=[GRID_Y_MIN, GRID_Y_MAX, GRID_X_MIN, GRID_X_MAX],
              origin="upper", interpolation="bilinear", zorder=1)

    # 3. Grid
    for v in range(int(GRID_Y_MIN), int(GRID_Y_MAX) + 1, 10):
        ax.axvline(v, color="#ffffff", alpha=0.07, linewidth=0.4, zorder=2)
    for v in range(int(GRID_X_MIN), int(GRID_X_MAX) + 1, 10):
        ax.axhline(v, color="#ffffff", alpha=0.07, linewidth=0.4, zorder=2)
    for r in [15, 30, 50]:
        ax.add_patch(plt.Circle((0, 0), r, color="#ffffff", fill=False,
                                linewidth=0.6, alpha=0.25, linestyle="--", zorder=2))
        ax.text(0.5, r + 0.8, f"{r} m", color="#cccccc", alpha=0.7,
                fontsize=8, ha="center", fontweight="bold", zorder=2)

    # 4. Camera FOV cones
    for cam in ALL_CAMS:
        cam_data  = nusc.get("sample_data", sample["data"][cam])
        calib_rec = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        color = CAM_COLORS[cam]
        px, py, heading = fov_cone_for_camera(calib_rec)
        ax.fill(px, py, color=color, alpha=0.06, zorder=3)
        ax.plot(px, py, color=color, alpha=0.8, linewidth=2.0, zorder=3)
        # Place label along camera axis, clamped to stay inside grid
        label_dist = 40
        lx_raw = np.cos(heading) * label_dist
        ly_raw = np.sin(heading) * label_dist
        lx = float(np.clip(lx_raw, GRID_Y_MIN + 2, GRID_Y_MAX - 2))
        ly = float(np.clip(ly_raw, GRID_X_MIN + 2, GRID_X_MAX - 2))
        ax.text(lx, ly, CAM_LABELS[cam], color=color, fontsize=7.5,
                fontweight="bold", ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0b0f18",
                          edgecolor=color, linewidth=0.9, alpha=0.9))

    # 5. GT boxes
    for gt in gt_boxes:
        loc, size, yaw, cat = gt["location"], gt["size"], gt["yaw"], gt["category_name"]
        x, y = loc[0], loc[1]
        if not (GRID_X_MIN <= x <= GRID_X_MAX and GRID_Y_MIN <= y <= GRID_Y_MAX):
            continue
        color = category_color(cat)
        lw = 1.8 if cat.startswith("vehicle.") else 1.0
        xs, ys = rotated_rect_corners(-y, x, size[0], size[1], -yaw)
        ax.add_patch(plt.Polygon(np.column_stack([xs, ys]), closed=True,
                                 edgecolor=color, facecolor="none",
                                 linewidth=lw, alpha=0.95, zorder=4))

    # 6. Detection dots + labels
    label_positions = []
    for x_m, y_m, _sigma, conf, cls_name in all_lifted:
        if not (GRID_X_MIN <= x_m <= GRID_X_MAX and GRID_Y_MIN <= y_m <= GRID_Y_MAX):
            continue
        plot_x = -y_m
        color = CONFIDENCE_CMAP(conf)
        ax.scatter(plot_x, x_m, c=[color[:3]], s=55, zorder=6,
                   edgecolors="white", linewidths=0.6)
        too_close = any(abs(plot_x - lx) < 3 and abs(x_m - ly) < 3
                        for lx, ly in label_positions)
        if not too_close:
            ax.annotate(cls_name, xy=(plot_x, x_m),
                        xytext=(plot_x + 1.3, x_m + 1.3),
                        color="white", fontsize=6.5, fontweight="bold", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#0b0f18",
                                  edgecolor="none", alpha=0.65))
            label_positions.append((plot_x, x_m))

    # 7. Ego vehicle
    ax.fill([-1, 1, 1, -1, -1], [-1.2, -1.2, 2.5, 2.5, -1.2],
            color="#ffd600", alpha=0.95, zorder=7)

    # 8. Colorbar
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Detection Confidence", color="#aaaaaa", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa")

    # 9. Legend
    ax.legend(handles=[
        mpatches.Patch(facecolor="#4fc3f7", edgecolor="#4fc3f7", alpha=0.8, label="GT vehicle"),
        mpatches.Patch(facecolor="#ff8a65", edgecolor="#ff8a65", alpha=0.8, label="GT pedestrian"),
        mpatches.Patch(facecolor="#ffd600", label="Ego vehicle"),
    ], loc="upper right", facecolor="#111827", labelcolor="#dddddd",
       fontsize=8, framealpha=0.88)

    ax.set_xlim(GRID_Y_MIN, GRID_Y_MAX)
    ax.set_ylim(GRID_X_MIN, GRID_X_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("← left  |  right →  (m)", color="#cccccc", fontsize=10)
    ax.set_ylabel("Forward (m)",              color="#cccccc", fontsize=10)
    ax.tick_params(colors="#cccccc", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444455")

    ax.set_title("BEV Confidence Heatmap", color="#eeeeee",
                 fontsize=14, fontweight="bold", pad=6)
    ax.text(0.5, 1.005, scene_name, transform=ax.transAxes,
            ha="center", va="bottom", color="#777777", fontsize=9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0b0f18")
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
    print(f"Scene: {scene['name']} — {scene['description']}")

    render(nusc, token, args.model, args.output,
           scene_name=scene["name"], num_passes=args.num_passes)


if __name__ == "__main__":
    main()
