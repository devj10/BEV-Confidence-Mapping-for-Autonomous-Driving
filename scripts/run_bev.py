"""
run_bev.py — top-level BEV pipeline conductor.

Wires together: detector → lift (pre-pass) → splat → BevFrame.

Public API:
    run_bev_frame()      — generic conductor; accepts swappable fn slots.
    run_bev_frame_real() — full pipeline; lifts raw dicts via GT-depth, then splats.
    run_bev_on_test_scene() — self-contained sanity check; no nuScenes needed.

Swap guide when real components land:
    detector_fn  → function returning real detection dicts
    lift_frame_gt / lift_frame_lidar → already wired; swap at call site
    passes_fn    → real per-pass MC lifting (replace scatter_passes)
    MODE / N_PASSES → edit in splat.py; run_bev inherits automatically.
"""

from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np

# bev/ modules use direct imports, so bev/ must be on sys.path
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "bev"))

from splat import (
    splat_detections,
    all_t_splat,
    render,
    MODE,
    N_PASSES,
    GRID_H, GRID_W, CELL,
    X_MIN, X_MAX, Y_MIN, Y_MAX,
)
from lift_adapter import (
    LiftedDetection,
    lifted_lift_fn,
    lift_frame_gt,
    lift_frame_lidar,
    scatter_passes,
)
from lift_to_3d import lift_to_3d  # exported for callers who need the trivial shim


# ---------------------------------------------------------------------------
# BevFrame — the return package
# ---------------------------------------------------------------------------

@dataclass
class BevFrame:
    """Everything produced by one run of the BEV pipeline."""

    grid:       np.ndarray       # float32 (GRID_H, GRID_W) heat map
    detections: list[dict]       # one dict per object: cls, score, bev_x, bev_y
    mode:       str              # "single_box" | "all_t"
    n_passes:   int

    @property
    def grid_spec(self) -> dict:
        return {
            "H": GRID_H, "W": GRID_W, "cell_m": CELL,
            "x_range": (X_MIN, X_MAX),
            "y_range": (Y_MIN, Y_MAX),
        }

    def to_dict(self) -> dict:
        """Flatten to plain Python — ready for export_bev.py to serialise."""
        return {
            "mode":       self.mode,
            "n_passes":   self.n_passes,
            "grid_spec":  self.grid_spec,
            "grid":       self.grid.tolist(),
            "detections": self.detections,
        }


# ---------------------------------------------------------------------------
# _bev_coords — "where does the marker go?"
# ---------------------------------------------------------------------------

def _bev_coords(
    det: LiftedDetection,
    passes_fn,
    mode: str,
    n_passes: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return BEV (x_m, y_m) centre for one already-lifted detection.

    single_box: exact position from the LiftedDetection.
    all_t:      mean of the n_passes scattered positions (expected location).
    """
    if mode == "single_box":
        return det.x_m, det.y_m
    cloud = passes_fn(det, n_passes, rng)      # (n_passes, 2)
    return float(cloud[:, 0].mean()), float(cloud[:, 1].mean())


# ---------------------------------------------------------------------------
# run_bev_frame — generic conductor (swappable slots)
# ---------------------------------------------------------------------------

def run_bev_frame(
    detector_fn,
    lift_fn,
    passes_fn,
    mode: str = MODE,
    n_passes: int = N_PASSES,
    rng: np.random.Generator | None = None,
) -> BevFrame:
    """Run the BEV pipeline for one frame.

    Args:
        detector_fn — callable() -> list[LiftedDetection]
        lift_fn     — callable(det) -> (x_m, y_m, sigma_x, sigma_y)
                      Use lifted_lift_fn when detections are LiftedDetection objects.
        passes_fn   — callable(det, n, rng) -> (n, 2) array of (x_m, y_m)
                      Used only in all_t mode.
        mode        — "single_box" | "all_t"
        n_passes    — MC passes (all_t only)
        rng         — optional seeded Generator for reproducibility

    Returns:
        BevFrame with the accumulated grid and per-detection metadata.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    lifted: list[LiftedDetection] = detector_fn()

    if mode == "single_box":
        grid = splat_detections(lifted, lift_fn=lift_fn)
    elif mode == "all_t":
        _passes = lambda det, n: passes_fn(det, n, rng)
        grid = all_t_splat(lifted, passes_fn=_passes, n_passes=n_passes)
    else:
        raise ValueError(f"unknown mode={mode!r}")

    det_records = []
    for det in lifted:
        bev_x, bev_y = _bev_coords(det, passes_fn, mode, n_passes, rng)
        det_records.append({
            "cls":   det.cls,
            "score": float(det.score),
            "bev_x": round(bev_x, 3),
            "bev_y": round(bev_y, 3),
        })

    return BevFrame(grid=grid, detections=det_records, mode=mode, n_passes=n_passes)


# ---------------------------------------------------------------------------
# run_bev_frame_real — full pipeline with lift pre-pass (GT-depth)
# ---------------------------------------------------------------------------

def run_bev_frame_real(
    raw_dets: list[dict],
    gt_boxes: list[dict],
    calib: dict,
    ego_pose: dict,
    mode: str = MODE,
    n_passes: int = N_PASSES,
    rng: np.random.Generator | None = None,
) -> BevFrame:
    """Lift raw detection dicts → LiftedDetection, then splat.

    This is the production entry point. Swap lift_frame_gt for lift_frame_lidar
    when the LiDAR-depth path is ready; nothing else changes.
    """
    lifted = lift_frame_gt(raw_dets, gt_boxes, calib, ego_pose)
    return run_bev_frame(
        detector_fn=lambda: lifted,
        lift_fn=lifted_lift_fn,
        passes_fn=scatter_passes,
        mode=mode,
        n_passes=n_passes,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# run_bev_on_test_scene — self-contained sanity check, no nuScenes needed
# ---------------------------------------------------------------------------

_TEST_SCENE: list[LiftedDetection] = [
    # near & confident → tight blobs
    LiftedDetection(x_m=5.0,  y_m=0.0,   sigma_x=0.20, sigma_y=0.20, score=0.93, cls="car"),
    LiftedDetection(x_m=8.0,  y_m=3.0,   sigma_x=0.25, sigma_y=0.22, score=0.88, cls="pedestrian"),
    # mid range → medium blobs
    LiftedDetection(x_m=20.0, y_m=-8.0,  sigma_x=0.80, sigma_y=0.50, score=0.81, cls="car"),
    LiftedDetection(x_m=25.0, y_m=6.0,   sigma_x=1.00, sigma_y=0.60, score=0.76, cls="truck"),
    # far → wide soft blobs (sigma_x >> sigma_y: depth dominates)
    LiftedDetection(x_m=42.0, y_m=10.0,  sigma_x=2.50, sigma_y=1.20, score=0.61, cls="car"),
    LiftedDetection(x_m=46.0, y_m=-15.0, sigma_x=3.20, sigma_y=1.50, score=0.55, cls="truck"),
    LiftedDetection(x_m=38.0, y_m=-2.0,  sigma_x=2.00, sigma_y=0.90, score=0.58, cls="bicycle"),
]


def run_bev_on_test_scene(
    mode: str = MODE,
    n_passes: int = N_PASSES,
    rng: np.random.Generator | None = None,
) -> BevFrame:
    """Run run_bev_frame with hard-coded LiftedDetection objects.

    No nuScenes, no fixture deps. Covers the full splat + adapter path.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    return run_bev_frame(
        detector_fn=lambda: list(_TEST_SCENE),
        lift_fn=lifted_lift_fn,
        passes_fn=scatter_passes,
        mode=mode,
        n_passes=n_passes,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# __main__ — self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for mode in ("single_box", "all_t"):
        frame = run_bev_on_test_scene(mode=mode)

        out = f"run_bev_{mode}.png"
        render(frame.grid, path=out, title=f"run_bev — {mode}")

        spec = frame.grid_spec
        print(f"\n{'='*56}")
        print(f"mode={mode}  grid={spec['H']}x{spec['W']}  "
              f"cell={spec['cell_m']} m  n_passes={frame.n_passes}")
        print(f"  grid max={frame.grid.max():.4f}  "
              f"nonzero={int((frame.grid > 0).sum())} cells")
        print(f"  {'cls':11s} {'score':>6}  {'bev_x':>7}  {'bev_y':>7}")
        for d in frame.detections:
            print(f"  {d['cls']:11s} {d['score']:6.2f}  "
                  f"{d['bev_x']:7.2f}  {d['bev_y']:7.2f}")

        assert frame.grid.max() > 0,       f"[{mode}] grid is empty"
        assert len(frame.detections) == 7, f"[{mode}] expected 7 detections"

    print("\nAll assertions passed.")
