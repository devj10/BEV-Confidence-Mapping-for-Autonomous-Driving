"""
3D Lifting: Project 2D detections from camera to ego frame.

Contract:
  Input:  detection = {"xyxy": [x1, y1, x2, y2], "conf": ..., ...}
  Output: (x_m, y_m, sigma_x, sigma_y) in ego frame (x forward, y left, z up)
  
Modes:
  - GT-depth: Match detection to nearest GT box, use its depth
  - LiDAR-depth: Use robust LiDAR depth (see lidar_project.py)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix


def get_camera_intrinsics(calib: Dict) -> np.ndarray:
    """Extract 3×3 camera matrix K from nuScenes calibration."""
    return np.array(calib["camera_intrinsic"], dtype=np.float32)


def get_camera_extrinsics(calib: Dict, ego_pose: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract camera→ego transformation.
    
    Returns:
        (rotation_matrix, translation_vector) where:
        - 3×3 rotation from camera to ego
        - 3×1 translation in ego frame
    
    nuScenes calibrated_sensor stores the sensor→ego transform. This helper
    returns that transform directly so camera_to_ego(point) is R @ point + t.
    """
    _ = ego_pose
    q_sensor_ego = Quaternion(calib["rotation"])
    R_sensor_ego = np.array(q_sensor_ego.rotation_matrix, dtype=np.float32)
    t_sensor_ego = np.array(calib["translation"], dtype=np.float32).reshape(3, 1)
    return R_sensor_ego, t_sensor_ego


def get_frame_calibration(nusc, sample_token: str, camera_channel: str = "CAM_FRONT") -> Dict:
    """
    Pull calibration for one frame via nuScenes devkit.

    Returns sample_data, calibrated_sensor, ego_pose, K, and camera→ego matrix.
    """
    sample = nusc.get("sample", sample_token)
    camera_token = sample["data"][camera_channel]
    sample_data = nusc.get("sample_data", camera_token)
    calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    K = get_camera_intrinsics(calib)
    camera_to_ego = transform_matrix(
        calib["translation"],
        Quaternion(calib["rotation"]),
        inverse=False,
    )
    return {
        "sample_data": sample_data,
        "calibrated_sensor": calib,
        "ego_pose": ego_pose,
        "K": K,
        "camera_to_ego": camera_to_ego,
    }


def invert_rigid_transform(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return inverse transform for point_b = R @ point_a + t."""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def backproject_to_camera_frame(
    u: float, v: float, depth: float, K: np.ndarray
) -> np.ndarray:
    """
    Back-project pixel (u, v) at depth z to camera frame 3D point.
    
    Args:
        u, v: pixel coordinates
        depth: depth in meters
        K: 3×3 camera matrix
    
    Returns:
        3×1 point in camera frame [x_cam, y_cam, z_cam]^T
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth
    
    return np.array([[x_cam], [y_cam], [z_cam]], dtype=np.float32)


def camera_to_ego(point_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Transform point from camera frame to ego frame.
    
    Args:
        point_cam: 3×1 point in camera frame
        R: 3×3 rotation (camera → ego)
        t: 3×1 translation (camera origin in ego)
    
    Returns:
        3×1 point in ego frame
    """
    point_ego = R @ point_cam + t
    return point_ego


def detection_to_bottom_center_pixel(
    xyxy: List[float], image_height: int = 900
) -> Tuple[float, float]:
    """
    Convert bounding box [x1, y1, x2, y2] to bottom-center pixel.
    
    Args:
        xyxy: [x1, y1, x2, y2] from detection
        image_height: image height (default 900 for nuScenes)
    
    Returns:
        (u_bottom_center, v_bottom_center)
    """
    x1, y1, x2, y2 = xyxy
    u = (x1 + x2) / 2.0  # horizontal center
    v = y2  # bottom of box (highest y in image)
    return (u, v)


def find_nearest_gt_box(
    detection_xyxy: List[float],
    gt_boxes: List[Dict],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    detection_class_name: Optional[str] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Find nearest GT box to detection (by center distance in image).
    
    Args:
        detection_xyxy: [x1, y1, x2, y2] of detection
        gt_boxes: list of {"location": [x, y, z], ...}
        K, R, t: camera parameters
    
    Returns:
        (depth, x_m, y_m) of nearest GT box, or None if no match
    """
    if not gt_boxes:
        return None
    
    # Get detection center in image
    det_center_u = (detection_xyxy[0] + detection_xyxy[2]) / 2
    det_center_v = (detection_xyxy[1] + detection_xyxy[3]) / 2
    det_center = np.array([det_center_u, det_center_v])
    
    # For each GT box, project 3D center to image and find closest
    best_dist = float('inf')
    best_gt = None
    best_depth = None
    best_xy = None
    
    R_ego_cam, t_ego_cam = invert_rigid_transform(R, t)
    
    for gt in gt_boxes:
        if not gt_matches_detection_class(gt, detection_class_name):
            continue

        location = gt.get("location")
        if location is None:
            continue
        
        # 3D location in ego frame [x, y, z]
        point_ego = np.array(location).reshape(3, 1)
        
        # Project to camera frame
        point_cam = R_ego_cam @ point_ego + t_ego_cam
        
        # Check if point is in front of camera
        if point_cam[2, 0] <= 0.1:
            continue
        
        # Project to image
        point_img = K @ point_cam
        u_proj = point_img[0, 0] / point_img[2, 0]
        v_proj = point_img[1, 0] / point_img[2, 0]
        proj_point = np.array([u_proj, v_proj])
        
        # Check if projection is in image bounds
        if u_proj < 0 or u_proj > 1600 or v_proj < 0 or v_proj > 900:
            continue
        
        # Compute distance in image space
        dist = np.linalg.norm(proj_point - det_center)
        
        if dist < best_dist:
            best_dist = dist
            best_gt = gt
            best_depth = point_cam[2, 0]
            best_xy = (location[0], location[1])
    
    # Accept match if projection is reasonably close (within 100 pixels).
    if best_gt is None or best_dist > 100:
        return None
    
    return (float(best_depth), float(best_xy[0]), float(best_xy[1]))


def gt_matches_detection_class(gt: Dict, detection_class_name: Optional[str]) -> bool:
    """Map simple detector class names onto nuScenes category names."""
    if not detection_class_name:
        return True

    det_class = str(detection_class_name).lower()
    gt_category = str(gt.get("category_name", "")).lower()

    if det_class in {"car", "truck", "bus", "vehicle"}:
        return gt_category.startswith("vehicle.")
    if det_class in {"person", "pedestrian"}:
        return gt_category.startswith("human.pedestrian")
    if det_class in {"bicycle", "bike"}:
        return "bicycle" in gt_category
    if det_class in {"motorcycle", "motorbike"}:
        return "motorcycle" in gt_category

    return det_class in gt_category


def project_ego_point_to_image(
    point_ego: np.ndarray,
    K: np.ndarray,
    R_cam_ego: np.ndarray,
    t_cam_ego: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Project one ego-frame point to image pixels using camera→ego extrinsics."""
    point_ego = np.asarray(point_ego, dtype=np.float32).reshape(3, 1)
    R_ego_cam, t_ego_cam = invert_rigid_transform(R_cam_ego, t_cam_ego)
    point_cam = R_ego_cam @ point_ego + t_ego_cam
    depth = float(point_cam[2, 0])
    if depth <= 0.1:
        return None
    point_img = K @ point_cam
    u = float(point_img[0, 0] / point_img[2, 0])
    v = float(point_img[1, 0] / point_img[2, 0])
    return (u, v, depth)


def project_gt_box_centers_to_image(
    gt_boxes: List[Dict],
    calib: Dict,
    ego_pose: Dict,
    image_shape: Tuple[int, int] = (900, 1600),
    category_prefixes: Optional[Tuple[str, ...]] = None,
) -> List[Dict]:
    """
    Validation gate #1 helper: forward-project GT box centers onto the image.

    Returned dots should sit on cars when overlaid on the camera image.
    """
    K = get_camera_intrinsics(calib)
    R, t = get_camera_extrinsics(calib, ego_pose)
    image_h, image_w = image_shape
    projections = []

    for gt in gt_boxes:
        if category_prefixes is not None:
            category = str(gt.get("category_name", "")).lower()
            if not category.startswith(category_prefixes):
                continue

        location = gt.get("location")
        if location is None:
            continue
        projected = project_ego_point_to_image(location, K, R, t)
        if projected is None:
            continue
        u, v, depth = projected
        if 0 <= u < image_w and 0 <= v < image_h:
            projections.append({
                "gt": gt,
                "uv": (u, v),
                "depth": depth,
            })

    return projections


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IOU between two boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def lift_detection_to_3d_gt_depth(
    detection: Dict,
    gt_boxes: List[Dict],
    calib: Dict,
    ego_pose: Dict,
    image_height: int = 900,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Lift a single 2D detection to 3D ego frame using GT depth.
    
    Validation Gate #1: Forward-project GT box centers onto image;
                       dots must sit on cars.
    
    Args:
        detection: {"xyxy": [...], "conf": ..., ...}
        gt_boxes: list of GT boxes with location
        calib: calibrated_sensor from nuScenes
        ego_pose: ego_pose from nuScenes
        image_height: image height (900)
    
    Returns:
        (x_m, y_m, sigma_x, sigma_y) in ego frame, or None if failed
    """
    K = get_camera_intrinsics(calib)
    R, t = get_camera_extrinsics(calib, ego_pose)
    
    xyxy = detection["xyxy"]
    
    # Find nearest GT box (get depth)
    gt_result = find_nearest_gt_box(
        xyxy,
        gt_boxes,
        K,
        R,
        t,
        detection_class_name=detection.get("class_name"),
    )
    if gt_result is None:
        return None
    
    depth, x_gt, y_gt = gt_result
    
    # Get bottom-center pixel of detection
    u_bottom, v_bottom = detection_to_bottom_center_pixel(xyxy, image_height)
    
    # Back-project to camera frame
    point_cam = backproject_to_camera_frame(u_bottom, v_bottom, depth, K)
    
    # Transform to ego frame
    point_ego = camera_to_ego(point_cam, R, t)
    x_m = float(point_ego[0, 0])
    y_m = float(point_ego[1, 0])
    
    # Uncertainty in depth (GT is deterministic, so very low uncertainty)
    sigma_x = 0.1  # 10 cm along x
    sigma_y = 0.1  # 10 cm along y
    
    return (x_m, y_m, sigma_x, sigma_y)


def lift_to_3d(detection: Dict) -> Tuple[float, float, float, float]:
    """
    Contract shim for C/_fixtures.py:
    lift_to_3d(detection) -> (x_m, y_m, sigma_x, sigma_y) in ego frame.

    The detection must already contain a lifted ego-frame point, either as
    detection["lifted"] or top-level x_m/y_m/sigma_x/sigma_y fields.
    """
    lifted = detection.get("lifted", detection)
    return (
        float(lifted["x_m"]),
        float(lifted["y_m"]),
        float(lifted.get("sigma_x", 0.0)),
        float(lifted.get("sigma_y", 0.0)),
    )


def lift_detection_to_3d_with_depth(
    detection: Dict,
    depth_m: float,
    calib: Dict,
    ego_pose: Dict,
    var_z: Optional[float] = None,
    image_height: int = 900,
) -> Optional[Dict]:
    """
    Lift a detection with a provided camera-frame depth.

    Used by the LiDAR-depth path after robust per-box depth extraction.
    """
    if depth_m is None or depth_m <= 0:
        return None

    K = get_camera_intrinsics(calib)
    R, t = get_camera_extrinsics(calib, ego_pose)
    u_bottom, v_bottom = detection_to_bottom_center_pixel(
        detection["xyxy"], image_height
    )
    point_cam = backproject_to_camera_frame(u_bottom, v_bottom, depth_m, K)
    point_ego = camera_to_ego(point_cam, R, t)

    sigma_depth = float(np.sqrt(var_z)) if var_z is not None and var_z >= 0 else 0.0
    return {
        "x_m": float(point_ego[0, 0]),
        "y_m": float(point_ego[1, 0]),
        "sigma_x": sigma_depth,
        "sigma_y": sigma_depth,
        "var_z": None if var_z is None else float(var_z),
        "depth_m": float(depth_m),
    }


def lift_detections_batch_lidar_depth(
    detections: List[Dict],
    depth_results: List[Dict[str, Optional[float]]],
    calib: Dict,
    ego_pose: Dict,
) -> List[Optional[Dict]]:
    """Lift detections with LiDAR depth results containing depth_m and var_z."""
    results = []
    for detection, depth_result in zip(detections, depth_results):
        results.append(
            lift_detection_to_3d_with_depth(
                detection,
                depth_result.get("depth_m"),
                calib,
                ego_pose,
                var_z=depth_result.get("var_z"),
            )
        )
    return results


def lift_detections_batch_gt_depth(
    detections: List[Dict],
    gt_boxes: List[Dict],
    calib: Dict,
    ego_pose: Dict,
) -> List[Tuple[float, float, float, float]]:
    """
    Lift batch of detections using GT depth.
    
    Returns:
        List of (x_m, y_m, sigma_x, sigma_y), with None for failures
    """
    results = []
    for det in detections:
        result = lift_detection_to_3d_gt_depth(det, gt_boxes, calib, ego_pose)
        results.append(result)
    return results
