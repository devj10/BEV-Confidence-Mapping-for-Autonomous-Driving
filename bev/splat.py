"""
splat.py — BEV heat-map accumulator.

Single-box mode  (MODE = "single_box"):
    One 2D Gaussian per detection, sigma from lift_fn, weight = detector score.

All-T mode       (MODE = "all_t"):
    For each detection, lift all T MC-pass positions to BEV, splat a small
    Gaussian per pass, weight = score/T.  Summing T blobs gives the MC
    posterior: confident objects (tight pass cluster) → sharp peak;
    uncertain ones (spread cluster) → wide smear.

Switch modes by changing MODE below. Both paths call lift_fn the same way, so
swapping in Owner B's real lift_to_3d requires only one import change.

Grid convention (matches _fixtures.py / bev_grid.py / configs/default.yaml):
    rows  ← x (forward),  0 m … 50 m,   200 rows, 0.25 m/cell
    cols  ← y (lateral),  -25 m … 25 m,  200 cols, 0.25 m/cell
    row 0 = nearest (ego position), row 199 = farthest.
"""

from __future__ import annotations
import numpy as np

# --- grid constants come from bev_grid.py; aliased to keep splat's short names
from bev_grid import (
    GRID_X_MIN as X_MIN, GRID_X_MAX as X_MAX,
    GRID_Y_MIN as Y_MIN, GRID_Y_MAX as Y_MAX,
    CELL_SIZE  as CELL,
    GRID_HEIGHT as GRID_H,
    GRID_WIDTH  as GRID_W,
)

# ---------------------------------------------------------------------------
# "single_box" | "all_t"
MODE = "single_box"

N_PASSES = 20          # MC dropout passes (all-T mode)
PER_PASS_SIGMA = 1.0   # cells; aleatoric floor per pass blob (all-T mode)
# ---------------------------------------------------------------------------


# --- coordinate helpers -------------------------------------------------------

def world_to_cell(x_m: float, y_m: float) -> tuple[float, float]:
    """Ego-frame meters → fractional grid indices (row, col)."""
    row = (x_m - X_MIN) / CELL
    col = (y_m - Y_MIN) / CELL
    return row, col


def cell_to_world(row: float, col: float) -> tuple[float, float]:
    """Grid indices → ego-frame meters. Inverse of world_to_cell."""
    x_m = row * CELL + X_MIN
    y_m = col * CELL + Y_MIN
    return x_m, y_m


# --- core splatting -----------------------------------------------------------

def splat_gaussian(
    grid: np.ndarray,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    weight: float = 1.0,
) -> None:
    """Paint one 2D Gaussian blob onto grid in-place.

    cx, cy  — center in cells (float, need not be integer)
    sx, sy  — sigma in cells (floored at 1 so confident objects still render)
    weight  — peak amplitude (use detector score for single-box mode)
    Only touches cells within a ±3σ window, so it's O(σ²) not O(grid).
    """
    sx = max(sx, 1.0)
    sy = max(sy, 1.0)

    r0 = max(0,          int(cx - 3 * sx))
    r1 = min(grid.shape[0], int(cx + 3 * sx) + 1)
    c0 = max(0,          int(cy - 3 * sy))
    c1 = min(grid.shape[1], int(cy + 3 * sy) + 1)

    if r0 >= r1 or c0 >= c1:
        return

    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    blob = weight * np.exp(
        -0.5 * ((rr - cx) ** 2 / sx ** 2 + (cc - cy) ** 2 / sy ** 2)
    )
    grid[r0:r1, c0:c1] += blob


def splat_detections(
    detections,
    lift_fn,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Single-box mode: one blob per detection, weighted by detector score.

    detections — iterable of objects with a .score attribute
    lift_fn    — callable: det -> (x_m, y_m, sigma_x, sigma_y)
                 Use fake_lift_to_3d now; swap to real lift_to_3d when B lands.
    Returns the accumulated float32 grid (shape GRID_H x GRID_W).
    """
    if grid is None:
        grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    for det in detections:
        x_m, y_m, sigma_x, sigma_y = lift_fn(det)
        cx, cy = world_to_cell(x_m, y_m)
        sx = sigma_x / CELL
        sy = sigma_y / CELL
        splat_gaussian(grid, cx, cy, sx, sy, weight=float(det.score))

    return grid


def all_t_splat(
    detections,
    passes_fn,
    n_passes: int = N_PASSES,
    per_pass_sigma: float = PER_PASS_SIGMA,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """All-T mode: lift each of the T per-pass positions, splat each, sum.

    For every detection, calls passes_fn(det, n_passes) → (n_passes, 2) array
    of (x_m, y_m) BEV positions — one row per MC pass.  Each pass position is
    splatted as a small Gaussian (per_pass_sigma cells); the T blobs sum to the
    MC posterior for that detection.  Weight per pass = score / n_passes so the
    total per-detection weight equals the detector score (same scale as
    single-box mode).

    passes_fn swap point: use get_fake_passes now; real per-pass lift later.
    """
    if grid is None:
        grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    for det in detections:
        cloud = passes_fn(det, n_passes)          # (n_passes, 2)
        w = float(det.score) / n_passes
        for x_m, y_m in cloud:
            cx, cy = world_to_cell(x_m, y_m)
            splat_gaussian(grid, cx, cy,
                           sx=per_pass_sigma, sy=per_pass_sigma,
                           weight=w)

    return grid


# --- visualisation ------------------------------------------------------------

def render(grid: np.ndarray, path: str = "splat_test.png", title: str = "BEV heat map") -> None:
    """Save grid as a colour heat map.  Flips rows so ego (x=0) is at the bottom."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    display = np.flipud(grid)   # row 0 (near) → bottom of image

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        display,
        origin="upper",         # after flipud, row 0 of display = x=50 m → top
        extent=[Y_MIN, Y_MAX, X_MIN, X_MAX],   # (left, right, bottom, top)
        cmap="plasma",
        interpolation="bilinear",
    )
    ax.set_xlabel("y  lateral (m, +left)")
    ax.set_ylabel("x  forward (m)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="accumulated confidence")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


# --- smoke test ---------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from _fixtures import (
        get_fake_detections,
        fake_lift_to_3d,
        get_fake_passes,
    )

    dets = get_fake_detections()
    print(f"mode: {MODE}   ({len(dets)} detections, grid {GRID_H}x{GRID_W})\n")

    if MODE == "single_box":
        grid = splat_detections(dets, lift_fn=fake_lift_to_3d)
        out  = "splat_test.png"
        render(grid, out, title="single-box mode (fake fixtures)")

    elif MODE == "all_t":
        grid = all_t_splat(dets, passes_fn=get_fake_passes, n_passes=N_PASSES)
        out  = "splat_allt_test.png"
        render(grid, out, title=f"all-T mode — {N_PASSES} MC passes (fake fixtures)")

    else:
        raise ValueError(f"unknown MODE={MODE!r}, expected 'single_box' or 'all_t'")

    print(f"grid max: {grid.max():.4f}   nonzero cells: {np.count_nonzero(grid)}")
    for det in dets:
        x_m, y_m, *_ = fake_lift_to_3d(det)
        r, c = world_to_cell(x_m, y_m)
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < GRID_H and 0 <= ci < GRID_W:
            val = grid[ri, ci]
            print(f"  [{'OK' if val > 0 else 'MISS'}] {det.cls:11s} "
                  f"({x_m:5.1f} m, {y_m:6.1f} m) → cell ({ri:3d},{ci:3d})  "
                  f"peak≈{val:.4f}")
        else:
            print(f"  [SKIP] {det.cls} is outside the grid")

    assert grid.max() > 0, "grid is empty — something is wrong"
    print("\nAll assertions passed.")
