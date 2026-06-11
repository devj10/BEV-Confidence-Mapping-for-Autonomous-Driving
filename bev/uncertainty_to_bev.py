"""
uncertainty_to_bev.py — translate detector pixel uncertainty into BEV meters.

The detector measures uncertainty in pixels (how much the box center wobbled
across T MC passes). The BEV map is in meters. This file is the translator.

Physics in two lines:
    sigma_lat     (lateral / side-to-side): sigma_y = (z / fx) · √var_u
    sigma_forward (depth / forward):        sigma_x = √var_z

Both terms grow with distance:
    sigma_lat grows linearly with z — same pixel jitter means more meters farther out.
    sigma_forward grows with √var_z — LiDAR depth spread increases as points thin out.
    Result: far cars produce wide, elongated blobs; near cars produce tight dots.

The headline finding: forward uncertainty dominates. For a car at 30 m with
3-pixel jitter (var_u=9) and 0.8 m² depth variance, lateral sigma is ~0.07 m
but forward sigma is ~0.9 m — a 12× ratio. The camera pins down direction very
well; depth is the hard, uncertain part.

Inputs:
    var_u   — pixel² variance of box-center u across T passes (from scores.py)
    z       — depth in meters (from B's lift — the camera-frame z value)
    fx      — camera focal length in pixels (~1266 for nuScenes CAM_FRONT)
    var_z   — depth variance in m² from LiDAR spread (optional; see fallback)

Pipeline position:
    lift → apply_bev_sigma → splat
    apply_bev_sigma overwrites the placeholder sigma (0.1 m) from lift_to_3d
    with the real camera-physics-derived values before the blob is painted.
"""

from __future__ import annotations
import math
import os
import sys
from typing import Optional

import numpy as np

# nuScenes CAM_FRONT focal length (pixels). Override if using a different camera.
FX_NUSCENES = 1266.417


def pixel_var_to_bev_sigma(
    var_u: float,
    z: float,
    fx: float = FX_NUSCENES,
    var_z: Optional[float] = None,
) -> tuple[float, float]:
    """Convert pixel-level uncertainty to BEV sigma values in meters.

    Args:
        var_u   pixel² variance of box-center u-coordinate across T passes.
                Use compute_center_variance() from uncertainty/scores.py.
        z       Depth of the detection in meters (camera-frame z).
        fx      Camera focal length in pixels. Default: nuScenes CAM_FRONT.
        var_z   Depth variance in m², from LiDAR robust-depth spread (var_z key
                in lidar_project output). If None, falls back to lateral sigma
                so the blob is at least isotropic rather than flat.

    Returns:
        (sigma_x, sigma_y) in meters, where x=forward, y=lateral (ego frame).

    Formulas:
        sigma_y = (z / fx) * sqrt(var_u)
            — pixel jitter projected to ground-plane lateral offset.
              Same wobble means more meters when the car is farther away.
        sigma_x = sqrt(var_z)
            — depth spread maps directly to forward uncertainty.
              If var_z is unavailable, use sigma_y (isotropic fallback).
    """
    if z <= 0:
        raise ValueError(f"depth z must be positive, got {z}")
    if var_u < 0:
        raise ValueError(f"var_u must be non-negative, got {var_u}")

    sigma_y = (z / fx) * math.sqrt(var_u)    # lateral: pixel jitter → meters

    if var_z is not None and var_z >= 0:
        sigma_x = math.sqrt(var_z)            # forward: depth spread → meters
    else:
        sigma_x = sigma_y                     # isotropic fallback (no LiDAR)

    return sigma_x, sigma_y


def apply_bev_sigma(
    det,
    var_u: float,
    z: float,
    fx: float = FX_NUSCENES,
    var_z: Optional[float] = None,
) -> None:
    """Overwrite a detection's sigma fields with camera-physics-derived values.

    Works on any object with sigma_x / sigma_y attributes (LiftedDetection) or
    a dict with those keys. Mutates in place; returns None.

    Call this between lift and splat:
        lifted = lift_frame_gt(...)
        for det, z, var_u, var_z in zip(lifted, depths, vars_u, vars_z):
            apply_bev_sigma(det, var_u=var_u, z=z, var_z=var_z)
        grid = splat_detections(lifted, lift_fn=lifted_lift_fn)
    """
    sigma_x, sigma_y = pixel_var_to_bev_sigma(var_u, z, fx, var_z)

    if isinstance(det, dict):
        det["sigma_x"] = sigma_x
        det["sigma_y"] = sigma_y
    else:
        det.sigma_x = sigma_x
        det.sigma_y = sigma_y


# ---------------------------------------------------------------------------
# __main__ — formula verification + "both terms grow" proof + visual
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from splat import splat_gaussian, render, GRID_H, GRID_W, CELL, world_to_cell

    print("uncertainty_to_bev — pixel variance → BEV sigma\n")

    # 1. Headline: forward >> lateral at 30 m
    sx, sy = pixel_var_to_bev_sigma(var_u=9.0, z=30.0, var_z=0.8)
    print(f"Headline  z=30 m, var_u=9 px², var_z=0.8 m²")
    print(f"  sigma_forward (sigma_x) = {sx:.3f} m   [= √var_z]")
    print(f"  sigma_lat     (sigma_y) = {sy:.3f} m   [= (z/fx)·√var_u]")
    print(f"  ratio                   = {sx/sy:.1f}×  (forward dominates)\n")

    assert 0.8 < sx < 1.0,  f"forward sigma off: {sx:.3f}"
    assert sy  < 0.15,       f"lateral sigma off: {sy:.3f}"
    assert sx  > sy,         "forward must dominate"

    # 2. Both terms grow with distance — explicit monotonicity check
    RANGES  = [5.0, 10.0, 20.0, 30.0, 42.0, 50.0]
    VAR_U   = 9.0                                    # px²  — fixed detector jitter
    VAR_Z   = [0.04, 0.10, 0.40, 0.80, 1.60, 2.20]  # m²   — grows with range

    print(f"{'range':>7}  {'var_z':>7}  {'σ_fwd (m)':>10}  {'σ_lat (m)':>10}")
    print("-" * 42)
    rows = []
    for z, vz in zip(RANGES, VAR_Z):
        sxr, syr = pixel_var_to_bev_sigma(VAR_U, z, var_z=vz)
        rows.append((z, vz, sxr, syr))
        print(f"{z:>7.0f}  {vz:>7.2f}  {sxr:>10.3f}  {syr:>10.3f}")

    # Assert both grow monotonically with range
    fwds = [r[2] for r in rows]
    lats = [r[3] for r in rows]
    assert all(fwds[i] < fwds[i+1] for i in range(len(fwds)-1)), \
        "sigma_forward must grow with range"
    assert all(lats[i] < lats[i+1] for i in range(len(lats)-1)), \
        "sigma_lat must grow with range"
    print("\nBoth terms grow monotonically with distance  ✓")

    # 3. Fallback (no var_z)
    sx_fb, sy_fb = pixel_var_to_bev_sigma(VAR_U, 30.0, var_z=None)
    assert sx_fb == sy_fb, "fallback must be isotropic"
    print(f"Fallback (no var_z): sigma_x = sigma_y = {sx_fb:.3f} m  ✓\n")


    # 4. Visual: splat three blobs (near / mid / far) sized by real sigmas
    grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    cars = [
        (5.0,  0.0,  5.0,   0.04),   # near
        (25.0, 0.0,  25.0,  0.80),   # mid
        (45.0, 0.0,  45.0,  2.20),   # far
    ]
    for x_m, y_m, z, vz in cars:
        sx_c, sy_c = pixel_var_to_bev_sigma(VAR_U, z, var_z=vz)
        cx, cy = world_to_cell(x_m, y_m)
        splat_gaussian(grid, cx, cy,
                       sx=sx_c / CELL, sy=sy_c / CELL,
                       weight=1.0)

    render(grid, "uncertainty_widening.png",
           title="sigma grows with range — both terms\n"
                 "(same var_u; var_z rises from 0.04 → 2.20 m²)")
    print("Saved → uncertainty_widening.png")
    print("\nAll assertions passed.")
