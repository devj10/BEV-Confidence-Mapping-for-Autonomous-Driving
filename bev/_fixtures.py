"""
_fixtures.py — fake BEV detections so splat.py / run_bev.py can be built and
tested BEFORE Owner B's real lift_to_3d / lidar_project land.

What this stands in for
-----------------------
Owner B's `lift_to_3d(detection)` takes a 2D image detection and returns its
position on the BEV ground plane, in ego-frame METERS, plus a positional sigma:

        (x_m, y_m, sigma_x, sigma_y)

    x_m       forward distance ahead of the ego car  (ego frame: x forward)
    y_m       lateral offset, +left / -right          (ego frame: y left)
    sigma_x   forward (depth) positional uncertainty, in meters
    sigma_y   lateral positional uncertainty, in meters

These fixtures produce that same tuple for a handful of hand-placed objects,
plus a matching cloud of T per-pass samples for the all-T splat mode. Everything
downstream (uncertainty_to_bev, splat, run_bev, export_bev) can run on this with
zero dependency on B or on A's retrained checkpoint.

When B is done: delete the fixture import in run_bev.py and point it at the real
lift_to_3d. The tuple shape is identical, so nothing else changes.

Grid spec baked in below is the team-agreed one (matches bev_grid.py / config):
    x in [0, 50] m, y in [-25, 25] m, 0.25 m/cell -> 200 x 200 grid.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

# --- agreed grid spec (will live in configs/default.yaml + bev_grid.py; copied
#     here only so the fixtures + a quick test render are self-contained) -------
X_MIN, X_MAX = 0.0, 50.0      # forward, meters
Y_MIN, Y_MAX = -25.0, 25.0    # lateral, meters (+left)
CELL = 0.25                   # meters per cell
GRID_H = int(round((X_MAX - X_MIN) / CELL))   # 200  (rows  <- forward / x)
GRID_W = int(round((Y_MAX - Y_MIN) / CELL))   # 200  (cols  <- lateral / y)

_RNG = np.random.default_rng(0)   # seeded so the scene is reproducible


@dataclass
class FakeDetection:
    """One fake object in BEV. Mirrors the info a real detection carries."""
    x_m: float            # forward (ego x)
    y_m: float            # lateral (ego y, +left)
    sigma_x: float        # forward/depth uncertainty (m)
    sigma_y: float        # lateral uncertainty (m)
    cls: str              # class label, just for color/debug later
    score: float          # detector confidence (not the uncertainty)

    @property
    def range_m(self) -> float:
        return float(np.hypot(self.x_m, self.y_m))

    def as_tuple(self):
        """Exactly what B's lift_to_3d returns: (x_m, y_m, sigma_x, sigma_y)."""
        return (self.x_m, self.y_m, self.sigma_x, self.sigma_y)


# --- the fake scene ----------------------------------------------------------
# Hand-placed so you can eyeball it: a near car dead ahead, some mid-range
# objects, and far objects out near the 50 m edge. Sigmas grow with range, and
# the FORWARD sigma (sigma_x, depth-dominated) grows faster than lateral — this
# is the physical behavior B's real depth will reproduce, so your splat should
# already show fat blobs far away before B is even done.
_SCENE = [
    # near & confident -> tight blobs
    FakeDetection(x_m=5.0,  y_m=0.0,   sigma_x=0.20, sigma_y=0.20, cls="car",        score=0.93),
    FakeDetection(x_m=8.0,  y_m=3.0,   sigma_x=0.25, sigma_y=0.22, cls="pedestrian", score=0.88),
    # mid range -> medium blobs
    FakeDetection(x_m=20.0, y_m=-8.0,  sigma_x=0.80, sigma_y=0.50, cls="car",        score=0.81),
    FakeDetection(x_m=25.0, y_m=6.0,   sigma_x=1.00, sigma_y=0.60, cls="truck",      score=0.76),
    # far -> wide, soft blobs (depth uncertainty dominates -> sigma_x >> sigma_y)
    FakeDetection(x_m=42.0, y_m=10.0,  sigma_x=2.50, sigma_y=1.20, cls="car",        score=0.61),
    FakeDetection(x_m=46.0, y_m=-15.0, sigma_x=3.20, sigma_y=1.50, cls="truck",      score=0.55),
    FakeDetection(x_m=38.0, y_m=-2.0,  sigma_x=2.00, sigma_y=0.90, cls="bicycle",    score=0.58),
]


def get_fake_detections() -> list[FakeDetection]:
    """Aggregated fake detections — feed these to splat.py single-box mode."""
    return list(_SCENE)


def fake_lift_to_3d(detection: FakeDetection):
    """Drop-in stand-in for Owner B's real lift_to_3d.

    Real signature (agreed): lift_to_3d(detection) -> (x_m, y_m, sigma_x, sigma_y)
    so build splat / run_bev calling THIS, then swap the import when B lands.
    """
    return detection.as_tuple()


def get_fake_passes(detection: FakeDetection, n_passes: int = 20) -> np.ndarray:
    """Fake the T per-pass BEV points for ONE detection — feed to all-T mode.

    The real all-T splat lifts each of the T MC-pass detections to BEV; here we
    emulate that by scattering n_passes points around the mean with std equal to
    the detection's sigma. So a confident (small-sigma) object yields a tight
    cluster -> sharp peak, and an uncertain one sprays out -> wide smear, exactly
    like the real Monte-Carlo passes. Returns shape (n_passes, 2): columns = x, y.
    """
    xs = _RNG.normal(detection.x_m, detection.sigma_x, size=n_passes)
    ys = _RNG.normal(detection.y_m, detection.sigma_y, size=n_passes)
    return np.column_stack([xs, ys])


def get_fake_passes_all(n_passes: int = 20) -> list[np.ndarray]:
    """Per-pass clouds for the whole scene: list of (n_passes, 2) arrays."""
    return [get_fake_passes(d, n_passes) for d in _SCENE]


# --- tiny self-test: run `python _fixtures.py` to sanity-check the scene ------
if __name__ == "__main__":
    dets = get_fake_detections()
    print(f"grid: {GRID_H} x {GRID_W} cells "
          f"({X_MIN}-{X_MAX} m fwd, {Y_MIN}-{Y_MAX} m lat, {CELL} m/cell)\n")
    print(f"{'class':<11}{'range':>7}{'x_m':>7}{'y_m':>8}{'sig_x':>7}{'sig_y':>7}")
    for d in sorted(dets, key=lambda d: d.range_m):
        print(f"{d.cls:<11}{d.range_m:7.1f}{d.x_m:7.1f}{d.y_m:8.1f}"
              f"{d.sigma_x:7.2f}{d.sigma_y:7.2f}")

    # confirm the key property: sigma rises with range
    near = min(dets, key=lambda d: d.range_m)
    far  = max(dets, key=lambda d: d.range_m)
    assert far.sigma_x > near.sigma_x, "far object should be more uncertain!"
    print(f"\nOK: nearest sigma_x={near.sigma_x:.2f} < "
          f"farthest sigma_x={far.sigma_x:.2f}  (far = fuzzier, as intended)")

    # confirm per-pass clouds scatter by sigma
    cloud_near = get_fake_passes(near)
    cloud_far  = get_fake_passes(far)
    print(f"per-pass spread  near: x-std={cloud_near[:,0].std():.2f}  "
          f"far: x-std={cloud_far[:,0].std():.2f}  (far cloud spreads wider)")
