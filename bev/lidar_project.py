"""
LiDAR Depth Projection: Project LiDAR points onto image, extract depth per detection.

Provides robust depth computation (median, mean, or clustering) and depth uncertainty
(variance) for each detection box.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pyquaternion import Quaternion

from .lift_to_3d import invert_rigid_transform


def map_pointcloud_to_image(
    points_ego: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_shape: Tuple[int, int] = (900, 1600),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Project LiDAR points (ego frame) onto image plane.
    
    Args:
        points_ego: Nx3 array of points in ego frame [x, y, z]
        K: 3×3 camera matrix
        R: 3×3 rotation (camera → ego)
        t: 3×1 translation (camera origin in ego)
        image_shape: (height, width)
    
    Returns:
        (pixel_coords, depths) where:
        - pixel_coords: Nx2 [u, v] for points projected on image
        - depths: N array of depths for each point
        or (None, None) if projection fails
    """
    image_h, image_w = image_shape
    
    # Transform ego→camera from the camera-to-ego transform.
    R_ego_cam, t_ego_cam = invert_rigid_transform(R, t)
    
    points_cam = (R_ego_cam @ points_ego.T + t_ego_cam).T  # Nx3
    
    # Filter points in front of camera (z > 0)
    valid = points_cam[:, 2] > 0.1
    if not np.any(valid):
        return None, None
    
    points_cam = points_cam[valid]
    depths = points_cam[:, 2]
    
    # Project to image
    pixel_homo = (K @ points_cam.T).T  # Nx3
    pixel_coords = pixel_homo[:, :2] / pixel_homo[:, 2:3]  # Nx2
    
    # Filter points inside image bounds
    in_image = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_w) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_h)
    )
    
    if not np.any(in_image):
        return None, None
    
    return pixel_coords[in_image], depths[in_image]


def extract_depth_in_box(
    xyxy: List[float],
    pixel_coords: np.ndarray,
    depths: np.ndarray,
) -> Optional[List[float]]:
    """
    Extract all LiDAR depths inside a bounding box.
    
    Args:
        xyxy: [x1, y1, x2, y2]
        pixel_coords: Nx2 projected LiDAR pixels
        depths: N array of depths
    
    Returns:
        List of depths inside box, or empty list if none
    """
    x1, y1, x2, y2 = xyxy
    
    mask = (
        (pixel_coords[:, 0] >= x1) & (pixel_coords[:, 0] <= x2) &
        (pixel_coords[:, 1] >= y1) & (pixel_coords[:, 1] <= y2)
    )
    
    depths_in_box = depths[mask].tolist()
    return depths_in_box if depths_in_box else []


def compute_robust_depth(
    depths: List[float],
    method: str = "median",
    outlier_percentile: float = 95.0,
) -> Tuple[float, float]:
    """
    Compute robust depth estimate and variance.
    
    Args:
        depths: list of LiDAR depths in bounding box
        method: "median", "mean", or "cluster"
        outlier_percentile: percentile for outlier removal
    
    Returns:
        (depth_estimate, variance)
    """
    if not depths:
        return None, None
    
    depths = np.array(depths, dtype=np.float32)
    
    # Remove extreme outliers
    q_high = np.percentile(depths, outlier_percentile)
    depths = depths[depths <= q_high]
    
    if len(depths) == 0:
        return None, None
    
    if method == "median":
        depth_est = float(np.median(depths))
        variance = float(np.var(depths))
    
    elif method == "mean":
        depth_est = float(np.mean(depths))
        variance = float(np.var(depths))
    
    elif method == "cluster":
        # Find dominant cluster of depths
        depth_est, variance = _cluster_depths(depths)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return (depth_est, variance)


def _cluster_depths(depths: np.ndarray) -> Tuple[float, float]:
    """
    Find dominant cluster of depths using simple 1D clustering.
    
    Returns:
        (mean_depth, variance) of largest cluster
    """
    # Sort and group consecutive similar depths
    depths = np.sort(depths)
    diffs = np.diff(depths)
    gaps = np.where(diffs > 0.5)[0]  # 50cm threshold
    
    if len(gaps) == 0:
        # Single cluster
        return (float(np.mean(depths)), float(np.var(depths)))
    
    # Find largest cluster
    clusters = np.split(depths, gaps + 1)
    largest_cluster = max(clusters, key=len)
    
    return (float(np.mean(largest_cluster)), float(np.var(largest_cluster)))


def extract_depth_per_detection(
    detections: List[Dict],
    points_ego: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    method: str = "median",
) -> List[Tuple[Optional[float], Optional[float]]]:
    """
    Extract robust LiDAR depth for each detection.
    
    Args:
        detections: list of {"xyxy": [...], ...}
        points_ego: Nx3 LiDAR points in ego frame
        K, R, t: camera parameters
        method: depth computation method
    
    Returns:
        List of (depth_estimate, variance) tuples
    """
    # Project all LiDAR points
    pixel_coords, depths = map_pointcloud_to_image(points_ego, K, R, t)
    
    if pixel_coords is None:
        return [(None, None) for _ in detections]
    
    results = []
    for detection in detections:
        xyxy = detection["xyxy"]
        depths_in_box = extract_depth_in_box(xyxy, pixel_coords, depths)
        depth_est, variance = compute_robust_depth(depths_in_box, method=method)
        results.append((depth_est, variance))

    return results


def load_lidar_points_ego(nusc, sample_token: str) -> np.ndarray:
    """
    Load LiDAR_TOP points for a sample and transform them into ego frame.

    Returns:
        Nx3 float32 array of (x, y, z) in ego frame
    """
    from pyquaternion import Quaternion

    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    lidar_path = nusc.get_sample_data_path(lidar_token)

    # nuScenes LiDAR binary: (x, y, z, intensity, ring_index) float32
    raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    points = raw[:, :3]  # Nx3

    calib_lidar = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    R_lidar_ego = np.array(
        Quaternion(calib_lidar["rotation"]).rotation_matrix, dtype=np.float32
    )
    t_lidar_ego = np.array(calib_lidar["translation"], dtype=np.float32)

    return (R_lidar_ego @ points.T).T + t_lidar_ego  # Nx3


def map_sample_lidar_to_image(
    nusc,
    sample_token: str,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_shape: Tuple[int, int] = (900, 1600),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convenience wrapper: load sample LiDAR, project onto camera image.

    Returns:
        (pixel_coords Nx2, depths N) or (None, None)
    """
    points_ego = load_lidar_points_ego(nusc, sample_token)
    return map_pointcloud_to_image(points_ego, K, R, t, image_shape)


def extract_depth_per_detection_devkit(
    nusc,
    sample_token: str,
    detections: List[Dict],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    method: str = "median",
) -> List[Dict[str, Optional[float]]]:
    """
    Full devkit-integrated pipeline: load LiDAR, project, extract robust
    depth for each detection.

    Returns:
        List of {"depth_m": float | None, "var_z": float | None}
    """
    points_ego = load_lidar_points_ego(nusc, sample_token)
    raw = extract_depth_per_detection(detections, points_ego, K, R, t, method)
    return [{"depth_m": depth, "var_z": var} for depth, var in raw]

