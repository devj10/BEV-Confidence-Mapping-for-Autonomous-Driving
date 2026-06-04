#!/usr/bin/env python3
"""
associate.py

Match and cluster detections across T Monte Carlo passes.

The key challenge: YOLOv8 produces variable numbers of boxes per pass and with
variable confidence scores. We need to:
  1. Compute pairwise IoU between all detections across passes
  2. Build a bipartite graph (pass1 → pass2 → pass3 → ...)
  3. Cluster boxes that spatially overlap (IoU > threshold) and have the same class
  4. Return clusters: each cluster has a list of (pass_id, box_id) references

For MC-Dropout in detection, spatial matching is genuinely hard because unlike
classification, localization uncertainty is distributed. We cluster using IoU,
which is a reasonable proxy for "these are probably the same object."

This module exports:
  - associate_detections(all_detections, iou_thresh=0.5, class_match=True)
    Returns: clusters = [ [pass_id, box_idx, pass_id, box_idx, ...], ... ]
"""

import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import cdist


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] (xyxy) format.
    
    Args:
        box1, box2: shape (4,) with coords [x1, y1, x2, y2]
    
    Returns:
        float: IoU in [0, 1]
    """
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


def associate_detections(
    all_detections: List[List[dict]],
    iou_thresh: float = 0.5,
    class_match: bool = True,
) -> List[List[Tuple[int, int]]]:
    """
    Cluster detections across T passes using spatial IoU.
    
    Args:
        all_detections: List of T elements, each is a list of detections.
                        Each detection is a dict with keys:
                          - 'xyxy': (4,) array [x1, y1, x2, y2]
                          - 'class_id': int
                          - 'conf': float
        iou_thresh: IoU threshold above which boxes are considered matches
        class_match: If True, only match boxes with the same class_id
    
    Returns:
        clusters: List of clusters. Each cluster is a list of (pass_idx, det_idx)
                  tuples pointing to detections that belong together.
    """
    T = len(all_detections)
    if T == 0:
        return []
    if T == 1:
        # Single pass: each detection is its own cluster
        return [[(0, i)] for i in range(len(all_detections[0]))]

    # Use a Union-Find structure to group detections across passes
    # Flatten all detections into a single list with (pass_id, det_idx) labels
    uf = UnionFind(sum(len(dets) for dets in all_detections))

    # Build a mapping from flattened index to (pass_id, det_idx)
    idx_to_ref = []
    flat_idx = 0
    for pass_id in range(T):
        for det_idx in range(len(all_detections[pass_id])):
            idx_to_ref.append((pass_id, det_idx))
            flat_idx += 1

    # Match detections between consecutive passes
    for pass_a in range(T):
        for pass_b in range(pass_a + 1, T):
            dets_a = all_detections[pass_a]
            dets_b = all_detections[pass_b]

            for idx_a, det_a in enumerate(dets_a):
                for idx_b, det_b in enumerate(dets_b):
                    # Check class match if required
                    if class_match and det_a["class_id"] != det_b["class_id"]:
                        continue

                    # Compute IoU
                    iou = box_iou(det_a["xyxy"], det_b["xyxy"])

                    if iou > iou_thresh:
                        # Merge these two detections into the same cluster
                        flat_a = _flat_idx(pass_a, idx_a, all_detections)
                        flat_b = _flat_idx(pass_b, idx_b, all_detections)
                        uf.union(flat_a, flat_b)

    # Extract clusters from union-find
    clusters = {}
    for flat_idx, (pass_id, det_idx) in enumerate(idx_to_ref):
        root = uf.find(flat_idx)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append((pass_id, det_idx))

    return list(clusters.values())


def _flat_idx(pass_id: int, det_idx: int, all_detections: List[List[dict]]) -> int:
    """
    Convert (pass_id, det_idx) to a flattened index across all detections.
    """
    idx = 0
    for p in range(pass_id):
        idx += len(all_detections[p])
    idx += det_idx
    return idx


class UnionFind:
    """Simple union-find (disjoint-set) data structure."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
