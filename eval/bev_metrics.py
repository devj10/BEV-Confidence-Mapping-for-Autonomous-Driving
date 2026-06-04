"""
BEV Metrics: Evaluate 3D detection quality in bird's-eye view.

Metrics computed:
  - Position error (distance between detected and GT center in BEV)
  - Bucketed by range: [0-15m], [15-30m], [30-50m]
  - Recall, Precision, F1-score in BEV space
  - Average localization error per range bucket
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cdist


class BEVMetrics:
    """Compute BEV detection evaluation metrics."""
    
    def __init__(
        self,
        distance_thresh: float = 1.0,
        target_category_prefixes: Optional[Tuple[str, ...]] = (
            "vehicle.",
            "human.pedestrian",
        ),
    ):
        """
        Args:
            distance_thresh: Distance threshold for matching detection to GT (meters)
            target_category_prefixes: nuScenes category prefixes to evaluate.
                None evaluates every GT category.
        """
        self.distance_thresh = distance_thresh
        self.target_category_prefixes = target_category_prefixes
        self.range_buckets = [(0, 15), (15, 30), (30, 50)]  # meters
        
    def extract_positions(
        self, 
        detections: List[Tuple[float, float, float, float, bool]],
        gt_boxes: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract (x, y) positions from detections and GT.
        
        Args:
            detections: List of (x, y, sigma_x, sigma_y, success)
            gt_boxes: List of {"location": [x, y, z], ...}
        
        Returns:
            (det_positions, gt_positions) where each is Nx2 array
        """
        # Detections
        det_pos = []
        for x, y, sx, sy, success in detections:
            if success:
                det_pos.append([x, y])
        det_pos = np.array(det_pos) if det_pos else np.empty((0, 2))
        
        # GT boxes (filter to BEV range)
        gt_pos = []
        for gt in gt_boxes:
            if self.target_category_prefixes is not None:
                category = str(gt.get("category_name", "")).lower()
                if not category.startswith(self.target_category_prefixes):
                    continue
            loc = gt.get('location')
            if loc is None:
                continue
            x, y, z = loc[0], loc[1], loc[2]
            # Check if in BEV range [0, 50] x [-25, 25]
            if 0 <= x <= 50 and -25 <= y <= 25:
                gt_pos.append([x, y])
        gt_pos = np.array(gt_pos) if gt_pos else np.empty((0, 2))
        
        return det_pos, gt_pos
    
    def match_detections_to_gt(
        self,
        det_pos: np.ndarray,
        gt_pos: np.ndarray,
    ) -> List[Tuple[int, int, float]]:
        """
        Match detections to GT using distance-based matching.
        
        Args:
            det_pos: Nx2 detection positions
            gt_pos: Mx2 GT positions
        
        Returns:
            List of (det_idx, gt_idx, distance) for successful matches
        """
        if len(det_pos) == 0 or len(gt_pos) == 0:
            return []
        
        # Compute pairwise distances
        distances = cdist(det_pos, gt_pos, metric='euclidean')  # NxM
        
        matches = []
        used_det = set()
        used_gt = set()

        # Greedy matching: closest pair first; each det/GT used at most once
        flat_idx = np.argsort(distances.flatten())

        for idx in flat_idx:
            det_idx = idx // distances.shape[1]
            gt_idx = idx % distances.shape[1]
            dist = distances[det_idx, gt_idx]

            if dist <= self.distance_thresh and det_idx not in used_det and gt_idx not in used_gt:
                matches.append((det_idx, gt_idx, dist))
                used_det.add(det_idx)
                used_gt.add(gt_idx)
        
        return matches
    
    def compute_metrics_per_range(
        self,
        det_pos: np.ndarray,
        gt_pos: np.ndarray,
        matches: List[Tuple[int, int, float]],
    ) -> Dict[str, Dict]:
        """
        Compute metrics (recall, precision, error) bucketed by range.
        
        Args:
            det_pos: Nx2 detection positions
            gt_pos: Mx2 GT positions
            matches: List of (det_idx, gt_idx, distance)
        
        Returns:
            Dict with metrics per range bucket
        """
        results = {}
        
        for range_min, range_max in self.range_buckets:
            range_name = f"{range_min}-{range_max}m"
            
            # Filter GT to this range
            gt_in_range = gt_pos[
                (gt_pos[:, 0] >= range_min) & (gt_pos[:, 0] < range_max)
            ]
            num_gt = len(gt_in_range)
            
            # Filter detections to this range
            det_in_range = det_pos[
                (det_pos[:, 0] >= range_min) & (det_pos[:, 0] < range_max)
            ]
            num_det = len(det_in_range)
            
            # Get matches whose GT center is in this range.
            matched_in_range = [
                (d, g, dist) for d, g, dist in matches
                if (gt_pos[g, 0] >= range_min and gt_pos[g, 0] < range_max)
            ]
            num_matched = len(matched_in_range)
            
            # Compute metrics
            recall = num_matched / num_gt if num_gt > 0 else 0.0
            precision = num_matched / num_det if num_det > 0 else 0.0
            f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0
            
            # Average error
            errors = [dist for _, _, dist in matched_in_range]
            avg_error = np.mean(errors) if errors else 0.0
            median_error = np.median(errors) if errors else 0.0
            rmse_error = np.sqrt(np.mean(np.square(errors))) if errors else 0.0
            max_error = np.max(errors) if errors else 0.0
            
            results[range_name] = {
                'num_gt': int(num_gt),
                'num_det': int(num_det),
                'num_matched': int(num_matched),
                'recall': float(recall),
                'precision': float(precision),
                'f1': float(f1),
                'avg_error': float(avg_error),
                'median_error': float(median_error),
                'rmse_error': float(rmse_error),
                'max_error': float(max_error),
            }
        
        return results
    
    def compute_overall_metrics(
        self,
        det_pos: np.ndarray,
        gt_pos: np.ndarray,
        matches: List[Tuple[int, int, float]],
    ) -> Dict:
        """
        Compute overall metrics across all ranges.
        
        Returns:
            Dict with overall statistics
        """
        num_gt_total = len(gt_pos)
        num_det_total = len(det_pos)
        num_matched_total = len(matches)
        
        recall_total = num_matched_total / num_gt_total if num_gt_total > 0 else 0.0
        precision_total = num_matched_total / num_det_total if num_det_total > 0 else 0.0
        f1_total = 2 * (recall_total * precision_total) / (recall_total + precision_total) \
            if (recall_total + precision_total) > 0 else 0.0
        
        errors = [dist for _, _, dist in matches]
        avg_error_total = np.mean(errors) if errors else 0.0
        median_error = np.median(errors) if errors else 0.0
        
        return {
            'num_gt': int(num_gt_total),
            'num_det': int(num_det_total),
            'num_matched': int(num_matched_total),
            'recall': float(recall_total),
            'precision': float(precision_total),
            'f1': float(f1_total),
            'avg_error': float(avg_error_total),
            'median_error': float(median_error),
        }
    
    def evaluate(
        self,
        detections: List[Tuple[float, float, float, float, bool]],
        gt_boxes: List[Dict],
    ) -> Dict:
        """
        Full evaluation pipeline.
        
        Returns:
            Dict with 'overall', 'per_range', and 'matches' keys
        """
        # Extract positions
        det_pos, gt_pos = self.extract_positions(detections, gt_boxes)
        
        # Match
        matches = self.match_detections_to_gt(det_pos, gt_pos)
        
        # Compute metrics
        per_range = self.compute_metrics_per_range(det_pos, gt_pos, matches)
        overall = self.compute_overall_metrics(det_pos, gt_pos, matches)
        
        return {
            'overall': overall,
            'per_range': per_range,
            'matches': [(int(d), int(g), float(dist)) for d, g, dist in matches],
            'num_detections': len(det_pos),
            'num_gt': len(gt_pos),
        }


def print_metrics(metrics: Dict, title: str = "BEV METRICS"):
    """Pretty-print metrics."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    # Overall
    overall = metrics['overall']
    print("\n[OVERALL]")
    print(f"  GT objects:        {overall['num_gt']}")
    print(f"  Detections:        {overall['num_det']}")
    print(f"  Matched:           {overall['num_matched']}")
    print(f"  Recall:            {overall['recall']:.3f}")
    print(f"  Precision:         {overall['precision']:.3f}")
    print(f"  F1-Score:          {overall['f1']:.3f}")
    print(f"  Avg Error (m):     {overall['avg_error']:.3f}")
    print(f"  Median Error (m):  {overall['median_error']:.3f}")
    
    # Per-range
    print("\n[PER-RANGE BREAKDOWN]")
    print(f"{'Range':<10} {'GT':<6} {'Det':<6} {'Match':<6} {'Recall':<8} {'Precision':<10} {'F1':<8} {'Avg Error':<12}")
    print("-" * 80)
    
    for range_name, metrics_range in metrics['per_range'].items():
        print(
            f"{range_name:<10} "
            f"{metrics_range['num_gt']:<6} "
            f"{metrics_range['num_det']:<6} "
            f"{metrics_range['num_matched']:<6} "
            f"{metrics_range['recall']:<8.3f} "
            f"{metrics_range['precision']:<10.3f} "
            f"{metrics_range['f1']:<8.3f} "
            f"{metrics_range['avg_error']:<12.3f}"
        )
    
    print("=" * 80)
