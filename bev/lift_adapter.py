"""
lift_adapter.py — bridge between lift_to_3d batch functions and splat.py.

Turns raw detection dicts into LiftedDetection objects (ego-frame BEV coords)
so splat_detections / all_t_splat can consume them without caring about calib.
None results (lift failures) are filtered out here, so callers always get a
clean list of successfully-lifted objects.

Depth paths:
    lift_frame_gt    — GT-depth  (lift_detections_batch_gt_depth)
    lift_frame_lidar — LiDAR-depth (lift_detections_batch_lidar_depth)

Swap point: swap the depth path in run_bev_frame_real(); the splatting code
never changes.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from lift_to_3d import (
    lift_detections_batch_gt_depth,
    lift_detections_batch_lidar_depth,
)


# ---------------------------------------------------------------------------
# LiftedDetection — the splat-ready unit
# ---------------------------------------------------------------------------

@dataclass
class LiftedDetection:
    """Ego-frame BEV position + uncertainty + detector metadata.

    Has a .score attribute so splat_detections() can use it as weight, and
    x_m/y_m/sigma_x/sigma_y so lifted_lift_fn() can read them.
    """
    x_m:     float
    y_m:     float
    sigma_x: float
    sigma_y: float
    score:   float
    cls:     str


def lifted_lift_fn(det: LiftedDetection) -> tuple[float, float, float, float]:
    """Trivial reader — matches the lift_to_3d contract for splat_detections.

    Use this as lift_fn whenever detections are already LiftedDetection objects.
    """
    return (det.x_m, det.y_m, det.sigma_x, det.sigma_y)


# ---------------------------------------------------------------------------
# Batch lift + filter helpers
# ---------------------------------------------------------------------------

def _score(det: dict) -> float:
    return float(det.get("conf", det.get("score", 1.0)))

def _cls(det: dict) -> str:
    return str(det.get("class_name", det.get("cls", "unknown")))


def lift_frame_gt(
    detections: list[dict],
    gt_boxes: list[dict],
    calib: dict,
    ego_pose: dict,
) -> list[LiftedDetection]:
    """GT-depth path: batch-lift raw dicts → LiftedDetection, drop Nones."""
    raw = lift_detections_batch_gt_depth(detections, gt_boxes, calib, ego_pose)
    out: list[LiftedDetection] = []
    for det, result in zip(detections, raw):
        if result is None:
            continue
        x_m, y_m, sigma_x, sigma_y = result
        out.append(LiftedDetection(
            x_m=x_m, y_m=y_m,
            sigma_x=sigma_x, sigma_y=sigma_y,
            score=_score(det), cls=_cls(det),
        ))
    return out


def lift_frame_lidar(
    detections: list[dict],
    depth_results: list[dict],
    calib: dict,
    ego_pose: dict,
) -> list[LiftedDetection]:
    """LiDAR-depth path: batch-lift raw dicts → LiftedDetection, drop Nones."""
    raw = lift_detections_batch_lidar_depth(detections, depth_results, calib, ego_pose)
    out: list[LiftedDetection] = []
    for det, result in zip(detections, raw):
        if result is None:
            continue
        out.append(LiftedDetection(
            x_m=float(result["x_m"]),
            y_m=float(result["y_m"]),
            sigma_x=float(result["sigma_x"]),
            sigma_y=float(result["sigma_y"]),
            score=_score(det), cls=_cls(det),
        ))
    return out


# ---------------------------------------------------------------------------
# Per-pass scatter — stands in for real MC lifting in all-T mode
# ---------------------------------------------------------------------------

def scatter_passes(
    det: LiftedDetection,
    n_passes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_passes BEV positions around the detection's mean with its sigma.

    This mimics what real per-pass MC lifting produces: a tight cluster for
    confident/near objects (small sigma) and a wide spray for uncertain/far ones.
    Replace with real per-pass lift_fn output when MC inference lands.

    Returns shape (n_passes, 2): columns = [x_m, y_m].
    """
    xs = rng.normal(det.x_m, max(det.sigma_x, 1e-3), size=n_passes)
    ys = rng.normal(det.y_m, max(det.sigma_y, 1e-3), size=n_passes)
    return np.column_stack([xs, ys])
