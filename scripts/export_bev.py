#!/usr/bin/env python3
"""
export_bev.py — dump one BEV frame to JSON + PNG for the Three.js frontend.

Stub mode  (no args, no nuScenes):
    python scripts/export_bev.py
    Runs the built-in LiftedDetection test scene and exports immediately.
    Useful for wiring up the frontend before real data is ready.

Real mode  (requires nuScenes at data/v1.0-mini):
    python scripts/export_bev.py --sample-token <token> \
        [--depth-mode gt|lidar] [--mode single_box|all_t] [--out-dir <dir>]

Outputs written to --out-dir (default results/bev_export/):
    bev_frame.json  — grid + detections for Three.js consumption
    bev_frame.png   — green→red confidence heat map (confident=green, empty=red)

JSON schema:
    {
        "mode":        "single_box" | "all_t",
        "n_passes":    int,
        "grid_spec":   {"H": 200, "W": 200, "cell_m": 0.25,
                        "x_range": [0, 50], "y_range": [-25, 25]},
        "grid_max":    float,
        "grid_flat":   [float × H×W],   // row-major, normalised 0–1
        "detections":  [{"cls": str, "score": float,
                         "bev_x": float, "bev_y": float}, ...]
    }
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Make repo root, bev/, and scripts/ importable when run as a script.
# bev/ modules use direct imports; scripts/ is where run_bev now lives.
_REPO = Path(__file__).parent.parent
_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "bev"))
sys.path.insert(0, str(_SCRIPTS))

from run_bev import BevFrame, run_bev_on_test_scene
from splat import MODE, N_PASSES


# ---------------------------------------------------------------------------
# Colourmap — black background, red→yellow→green for confidence
# ---------------------------------------------------------------------------

_GREEN_RED = mcolors.LinearSegmentedColormap.from_list(
    "bev_confidence",
    [
        (0.00, "#0d0d0d"),   # 0.0  — empty cell, near-black
        (0.05, "#7f0000"),   # low  — faint red
        (0.35, "#cc3300"),   # mid-low
        (0.60, "#ffcc00"),   # mid-high — yellow
        (1.00, "#00e676"),   # 1.0  — confident, green
    ],
)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def frame_to_dict(frame: BevFrame) -> dict:
    """Serialise a BevFrame to a JSON-safe dict for the Three.js frontend."""
    grid = frame.grid
    gmax = float(grid.max()) if grid.max() > 0 else 1.0
    normalised = (grid / gmax).astype(np.float32)

    return {
        "mode":       frame.mode,
        "n_passes":   frame.n_passes,
        "grid_spec":  frame.grid_spec,
        "grid_max":   round(gmax, 6),
        "grid_flat":  [round(float(v), 5) for v in normalised.ravel()],
        "detections": frame.detections,
    }


def write_json(frame: BevFrame, path: Path) -> None:
    payload = frame_to_dict(frame)
    path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON → {path}  ({path.stat().st_size // 1024} KB, "
          f"{len(payload['detections'])} detections)")


def write_png(frame: BevFrame, path: Path) -> None:
    """Green→red confidence heat map. Ego is at the bottom (x=0)."""
    spec  = frame.grid_spec
    gmax  = frame.grid.max()
    norm  = frame.grid / gmax if gmax > 0 else frame.grid

    display = np.flipud(norm)   # row 0 (near) → bottom

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    im = ax.imshow(
        display,
        origin="upper",
        extent=[
            spec["y_range"][0], spec["y_range"][1],
            spec["x_range"][0], spec["x_range"][1],
        ],
        cmap=_GREEN_RED,
        vmin=0, vmax=1,
        interpolation="bilinear",
    )

    # Detection markers
    for d in frame.detections:
        ax.scatter(d["bev_y"], d["bev_x"],
                   marker="o", s=60, zorder=5,
                   edgecolors="white", linewidths=0.8,
                   c=[_GREEN_RED(d["score"])],
                   label=d["cls"])
        ax.text(d["bev_y"] + 0.5, d["bev_x"] + 0.5,
                f"{d['cls'][0].upper()}", color="white",
                fontsize=6, zorder=6, alpha=0.8)

    # Ego marker
    ax.scatter(0, 0, marker="^", s=120, c="#ffd600",
               edgecolors="white", linewidths=1, zorder=7)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("confidence (normalised)", color="#cccccc", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc")

    ax.set_xlabel("y  lateral (m, +left)", color="#cccccc", fontsize=9)
    ax.set_ylabel("x  forward (m)",        color="#cccccc", fontsize=9)
    ax.tick_params(colors="#888888", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")

    ax.set_title(
        f"BEV confidence — {frame.mode}  (T={frame.n_passes})",
        color="#eeeeee", fontsize=10, pad=8,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  PNG  → {path}")


def export_frame(frame: BevFrame, out_dir: Path, stem: str = "bev_frame") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(frame, out_dir / f"{stem}.json")
    write_png(frame,  out_dir / f"{stem}.png")


# ---------------------------------------------------------------------------
# Real-lift path (requires nuScenes)
# ---------------------------------------------------------------------------

def run_real(sample_token: str, depth_mode: str, bev_mode: str, n_passes: int):
    """Build a BevFrame from actual nuScenes data."""
    from nuscenes.nuscenes import NuScenes
    from lift_to_3d import get_frame_calibration, get_camera_intrinsics, get_camera_extrinsics
    from lift_adapter import lift_frame_gt, lift_frame_lidar, lifted_lift_fn, scatter_passes
    from run_bev import run_bev_frame

    nusc = NuScenes(version="v1.0-mini", dataroot="data/v1.0-mini", verbose=False)
    frame_calib = get_frame_calibration(nusc, sample_token, "CAM_FRONT")
    calib    = frame_calib["calibrated_sensor"]
    ego_pose = frame_calib["ego_pose"]

    # Pull GT boxes for GT-depth mode
    from scripts.visualize_rgb_bev import get_gt_boxes_ego
    gt_boxes = get_gt_boxes_ego(nusc, sample_token, ego_pose)

    # Placeholder: real pipeline would load detector output from a JSON here.
    # For now, surface an empty list so the grid/JSON/PNG are still produced.
    raw_dets: list[dict] = []

    if depth_mode == "gt":
        lifted = lift_frame_gt(raw_dets, gt_boxes, calib, ego_pose)
    else:
        K = get_camera_intrinsics(calib)
        R, t = get_camera_extrinsics(calib, ego_pose)
        from bev.lidar_project import extract_depth_per_detection_devkit
        depth_results = extract_depth_per_detection_devkit(
            nusc, sample_token, raw_dets, K, R, t
        )
        lifted = lift_frame_lidar(raw_dets, depth_results, calib, ego_pose)

    rng = np.random.default_rng(0)
    frame = run_bev_frame(
        detector_fn=lambda: lifted,
        lift_fn=lifted_lift_fn,
        passes_fn=scatter_passes,
        mode=bev_mode,
        n_passes=n_passes,
        rng=rng,
    )
    return frame


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export one BEV frame to JSON + PNG.")
    parser.add_argument("--sample-token", default=None,
                        help="nuScenes sample token. Omit for stub mode.")
    parser.add_argument("--depth-mode", default="gt", choices=["gt", "lidar"])
    parser.add_argument("--mode", default=MODE, choices=["single_box", "all_t"])
    parser.add_argument("--n-passes", type=int, default=N_PASSES)
    parser.add_argument("--out-dir", default="results/bev_export")
    parser.add_argument("--stem", default="bev_frame",
                        help="Output filename stem (without extension).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.sample_token is None:
        print("Stub mode — using test scene (no nuScenes required)")
        frame = run_bev_on_test_scene(mode=args.mode, n_passes=args.n_passes)
    else:
        print(f"Real mode — sample {args.sample_token[:16]}...  "
              f"depth={args.depth_mode}  bev={args.mode}")
        frame = run_real(args.sample_token, args.depth_mode, args.mode, args.n_passes)

    spec = frame.grid_spec
    print(f"Grid {spec['H']}×{spec['W']}  cell={spec['cell_m']} m  "
          f"max={frame.grid.max():.4f}  dets={len(frame.detections)}")

    export_frame(frame, out_dir, stem=args.stem)
    print("Done.")


if __name__ == "__main__":
    main()
