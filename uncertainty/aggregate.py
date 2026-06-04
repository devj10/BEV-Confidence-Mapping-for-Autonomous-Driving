#!/usr/bin/env python3
"""
aggregate.py

Given clusters of detections from associate.py, compute:
  - Mean box (x1, y1, x2, y2) per cluster
  - Center variance as uncertainty score
  - NMS to remove overlapping merged boxes

This module exports:
  - aggregate_clusters(clusters, all_detections, nms_thresh=0.5)
    Returns: merged_boxes = [ {"xyxy": ..., "class_id": ..., "score": ..., "uncertainty": ...}, ... ]
"""

import numpy as np
from typing import List, Tuple, Dict, Any


def aggregate_clusters(
    clusters: List[List[Tuple[int, int]]],
    all_detections: List[List[Dict[str, Any]]],
    nms_thresh: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Convert clusters of detections into merged boxes with uncertainty.
    
    Args:
        clusters: List of clusters from associate_detections().
                  Each cluster is a list of (pass_id, det_idx) tuples.
        all_detections: The same list passed to associate_detections().
        nms_thresh: IoU threshold for NMS on merged boxes.
    
    Returns:
        merged_boxes: List of dicts with keys:
          - 'xyxy': (4,) mean box coordinates [x1, y1, x2, y2]
          - 'class_id': int (from first detection in cluster)
          - 'conf': float (mean confidence across cluster)
          - 'uncertainty': float (normalized center variance)
          - 'num_detections': int (how many passes contributed to cluster)
    """
    merged_boxes = []

    for cluster in clusters:
        # Extract all boxes in this cluster
        boxes = []
        confs = []
        class_id = None

        for pass_id, det_idx in cluster:
            det = all_detections[pass_id][det_idx]
            boxes.append(det["xyxy"])
            confs.append(det.get("conf", 1.0))
            if class_id is None:
                class_id = det["class_id"]

        boxes = np.array(boxes)  # (N, 4)
        confs = np.array(confs)  # (N,)

        # Compute mean box
        mean_box = boxes.mean(axis=0)

        # Compute center as (x_c, y_c) = ((x1 + x2)/2, (y1 + y2)/2)
        centers = _box_centers(boxes)  # (N, 2)
        mean_center = centers.mean(axis=0)  # (2,)

        # Center variance: average squared distance from mean center
        center_diffs = centers - mean_center  # (N, 2)
        center_var = (center_diffs ** 2).mean()  # scalar
        
        # Normalize variance by image area (assume 640x640 image; adjust if needed)
        img_size = 640
        center_var_normalized = center_var / (img_size ** 2)

        # Mean confidence
        mean_conf = confs.mean()

        merged_boxes.append({
            "xyxy": mean_box,
            "class_id": class_id,
            "conf": mean_conf,
            "uncertainty": float(center_var_normalized),
            "num_detections": len(cluster),
        })

    # Apply NMS on merged boxes
    merged_boxes = _nms(merged_boxes, nms_thresh)

    return merged_boxes


def _box_centers(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes [x1, y1, x2, y2] to centers [(x1+x2)/2, (y1+y2)/2].
    
    Args:
        boxes: shape (N, 4)
    
    Returns:
        centers: shape (N, 2)
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    ], axis=1)


def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    if xi_max < xi_min or yi_max < yi_min:
        return 0.0

    inter_area = (xi_max - xi_min) * (yi_max - yi_min)

    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area < 1e-6:
        return 0.0

    return inter_area / union_area


def _nms(boxes: List[Dict[str, Any]], iou_thresh: float = 0.5) -> List[Dict[str, Any]]:
    """
    Apply Non-Maximum Suppression on merged boxes.
    Keeps boxes with highest confidence, removes overlapping ones.
    
    Args:
        boxes: List of dicts with 'xyxy' and 'conf' keys.
        iou_thresh: IoU threshold for suppression.
    
    Returns:
        Filtered list of boxes.
    """
    if len(boxes) == 0:
        return []

    # Sort by confidence (descending)
    boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)

    keep = []
    while len(boxes) > 0:
        keep.append(boxes[0])
        if len(boxes) == 1:
            break

        current_xyxy = boxes[0]["xyxy"]
        boxes = boxes[1:]

        # Remove boxes with high IoU to current box
        remaining = []
        for box in boxes:
            iou = _box_iou(current_xyxy, box["xyxy"])
            if iou <= iou_thresh:
                remaining.append(box)

        boxes = remaining

    return keep
