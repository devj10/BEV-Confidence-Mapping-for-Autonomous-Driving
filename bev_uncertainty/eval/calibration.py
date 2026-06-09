#!/usr/bin/env python3
"""
calibration.py

Phase 4 Evaluation: Uncertainty calibration and analysis.

Implements:
  1. Sparsification: remove high-uncertainty boxes, plot improvement in mAP
  2. Uncertainty-vs-Error correlation: measure if uncertainty predicts localization error
  3. Out-of-distribution detection: uncertainty rises on ACDC adverse weather (stretch)

Usage:
    from bev_uncertainty.eval.calibration import sparsification_analysis, uncertainty_error_correlation
    
    # After running MC inference on val set (ground truth + predictions)
    results = sparsification_analysis(
        detections_with_uncertainty,
        ground_truth_boxes,
        uncertainty_thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    
    correlation = uncertainty_error_correlation(
        detections_with_uncertainty,
        ground_truth_boxes,
        num_bins=10
    )
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class CalibrationMetrics:
    """Container for calibration analysis results."""
    uncertainty_threshold: float
    num_boxes_kept: int
    num_boxes_removed: int
    map_at_threshold: float
    map_improvement: float  # vs. all boxes
    precision: float
    recall: float


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in xyxy format.
    box: [x1, y1, x2, y2]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def match_detections_to_ground_truth(
    detections: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_thresh: float = 0.5,
    class_agnostic: bool = False
) -> Tuple[List[bool], List[float]]:
    """
    Match predicted detections to ground truth boxes.
    
    Args:
        detections: list of {"xyxy": [...], "class_id": int, "conf": float, "uncertainty": float}
        ground_truth: list of {"xyxy": [...], "class_id": int}
        iou_thresh: IoU threshold for a match (default 0.5)
        class_agnostic: if True, match boxes regardless of class
    
    Returns:
        is_matched: list of bools (True if detection matched to GT)
        ious: list of IoU values for each detection
    """
    matched = np.zeros(len(detections), dtype=bool)
    iou_per_det = np.zeros(len(detections), dtype=float)
    used_gt = set()
    
    # Sort detections by confidence (descending)
    sorted_idx = np.argsort([-d["conf"] for d in detections])
    
    for det_idx in sorted_idx:
        det = detections[det_idx]
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in used_gt:
                continue
            
            # Check class match
            if not class_agnostic and det["class_id"] != gt["class_id"]:
                continue
            
            iou = compute_iou(np.array(det["xyxy"]), np.array(gt["xyxy"]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched[det_idx] = True
            used_gt.add(best_gt_idx)
            iou_per_det[det_idx] = best_iou
    
    return matched.tolist(), iou_per_det.tolist()


def localization_error(
    det_box: np.ndarray,
    gt_box: np.ndarray
) -> float:
    """
    Compute localization error as 1 - IoU.
    Lower is better (0 = perfect match, 1 = no overlap).
    """
    iou = compute_iou(det_box, gt_box)
    return 1.0 - iou


def sparsification_analysis(
    detections_with_uncertainty: List[List[Dict[str, Any]]],
    ground_truth_per_frame: List[List[Dict[str, Any]]],
    uncertainty_thresholds: List[float] = None,
    iou_thresh: float = 0.5,
    class_agnostic: bool = False
) -> Dict[float, CalibrationMetrics]:
    """
    Sparsification: measure how removing high-uncertainty boxes affects mAP.
    
    Hypothesis: if uncertainty is well-calibrated, removing high-uncertainty 
    boxes should initially hurt (fewer boxes), then help once we remove enough false positives.
    
    Args:
        detections_with_uncertainty: list of frames, each frame has list of detections
        ground_truth_per_frame: same structure for GT
        uncertainty_thresholds: thresholds to test (keep boxes with unc < thresh)
        iou_thresh: IoU threshold for matching
        class_agnostic: ignore class when matching
    
    Returns:
        dict mapping threshold → CalibrationMetrics
    """
    if uncertainty_thresholds is None:
        uncertainty_thresholds = np.linspace(0.0, 1.0, 11).tolist()
    
    results = {}
    
    # Baseline: use all boxes (threshold = inf)
    all_matched = 0
    all_tp = 0
    all_fp = 0
    all_fn = 0
    
    for frame_idx, (frame_dets, frame_gt) in enumerate(
        zip(detections_with_uncertainty, ground_truth_per_frame)
    ):
        matched, _ = match_detections_to_ground_truth(
            frame_dets, frame_gt, iou_thresh=iou_thresh, class_agnostic=class_agnostic
        )
        tp = sum(matched)
        fp = len(matched) - tp
        fn = len(frame_gt) - tp
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_matched += 1
    
    baseline_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    baseline_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    baseline_map = (baseline_precision * baseline_recall) if baseline_recall > 0 else 0.0
    
    # Now test sparsification thresholds
    for threshold in sorted(uncertainty_thresholds):
        tp, fp, fn = 0, 0, 0
        
        for frame_idx, (frame_dets, frame_gt) in enumerate(
            zip(detections_with_uncertainty, ground_truth_per_frame)
        ):
            # Filter by uncertainty threshold
            filtered_dets = [d for d in frame_dets if d.get("uncertainty", 0.0) < threshold]
            
            matched, _ = match_detections_to_ground_truth(
                filtered_dets, frame_gt, iou_thresh=iou_thresh, class_agnostic=class_agnostic
            )
            tp += sum(matched)
            fp += len(matched) - sum(matched)
            fn += len(frame_gt) - sum(matched)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        map_at_thresh = (precision * recall) if recall > 0 else 0.0
        
        results[threshold] = CalibrationMetrics(
            uncertainty_threshold=threshold,
            num_boxes_kept=len([d for frame in detections_with_uncertainty for d in frame 
                               if d.get("uncertainty", 0.0) < threshold]),
            num_boxes_removed=len([d for frame in detections_with_uncertainty for d in frame 
                                  if d.get("uncertainty", 0.0) >= threshold]),
            map_at_threshold=map_at_thresh,
            map_improvement=map_at_thresh - baseline_map,
            precision=precision,
            recall=recall
        )
    
    return results


def uncertainty_error_correlation(
    detections_with_uncertainty: List[List[Dict[str, Any]]],
    ground_truth_per_frame: List[List[Dict[str, Any]]],
    num_bins: int = 10,
    iou_thresh: float = 0.5,
    class_agnostic: bool = False
) -> Dict[str, Any]:
    """
    Measure correlation between uncertainty and localization error.
    
    For each matched detection, compute:
      - Uncertainty (from MC dropout)
      - Localization error (1 - IoU with matched GT box)
    
    Then bin by uncertainty and compute average error per bin.
    
    Args:
        detections_with_uncertainty: list of frames with detections
        ground_truth_per_frame: corresponding GT
        num_bins: number of uncertainty bins for correlation analysis
        iou_thresh: IoU threshold for matching
        class_agnostic: ignore class when matching
    
    Returns:
        dict with keys:
          - uncertainty_bins: list of bin centers
          - mean_errors: mean error per bin
          - std_errors: std of error per bin
          - count_per_bin: number of detections per bin
          - spearman_corr: Spearman correlation coefficient
          - pearson_corr: Pearson correlation coefficient
    """
    uncertainties = []
    errors = []
    ious = []
    
    for frame_dets, frame_gt in zip(detections_with_uncertainty, ground_truth_per_frame):
        matched, iou_per_det = match_detections_to_ground_truth(
            frame_dets, frame_gt, iou_thresh=iou_thresh, class_agnostic=class_agnostic
        )
        
        # Only compute error for matched detections
        for det, is_matched, iou_val in zip(frame_dets, matched, iou_per_det):
            if is_matched:
                uncertainty = det.get("uncertainty", 0.0)
                error = 1.0 - iou_val  # localization error
                uncertainties.append(uncertainty)
                errors.append(error)
                ious.append(iou_val)
    
    if not uncertainties:
        return {
            "uncertainty_bins": [],
            "mean_errors": [],
            "std_errors": [],
            "count_per_bin": [],
            "spearman_corr": 0.0,
            "pearson_corr": 0.0,
            "mean_iou_per_bin": []
        }
    
    uncertainties = np.array(uncertainties)
    errors = np.array(errors)
    ious = np.array(ious)
    
    # Compute Pearson and Spearman correlations
    pearson_corr = np.corrcoef(uncertainties, errors)[0, 1] if len(uncertainties) > 1 else 0.0
    
    # Spearman: rank correlation
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(uncertainties, errors) if len(uncertainties) > 1 else (0.0, 1.0)
    
    # Bin by uncertainty
    bins = np.linspace(0, np.max(uncertainties) + 1e-6, num_bins + 1)
    bin_indices = np.digitize(uncertainties, bins) - 1
    
    bin_centers = []
    mean_errors_per_bin = []
    std_errors_per_bin = []
    counts_per_bin = []
    mean_ious_per_bin = []
    
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        if np.sum(mask) > 0:
            bin_centers.append(np.mean(uncertainties[mask]))
            mean_errors_per_bin.append(np.mean(errors[mask]))
            std_errors_per_bin.append(np.std(errors[mask]))
            counts_per_bin.append(np.sum(mask))
            mean_ious_per_bin.append(np.mean(ious[mask]))
    
    return {
        "uncertainty_bins": bin_centers,
        "mean_errors": mean_errors_per_bin,
        "std_errors": std_errors_per_bin,
        "count_per_bin": counts_per_bin,
        "mean_iou_per_bin": mean_ious_per_bin,
        "spearman_corr": float(spearman_corr),
        "pearson_corr": float(pearson_corr),
        "num_matched_detections": len(uncertainties)
    }


def save_calibration_report(
    sparsification_results: Dict[float, CalibrationMetrics],
    correlation_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Save calibration analysis to JSON and print summary.
    
    Args:
        sparsification_results: from sparsification_analysis()
        correlation_results: from uncertainty_error_correlation()
        output_dir: directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sparsification
    sparsif_dict = {
        str(k): asdict(v) for k, v in sparsification_results.items()
    }
    sparsif_path = output_dir / "sparsification.json"
    with open(sparsif_path, "w") as f:
        json.dump(sparsif_dict, f, indent=2)
    
    # Save correlation
    corr_path = output_dir / "correlation.json"
    with open(corr_path, "w") as f:
        json.dump(correlation_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("PHASE 4 — CALIBRATION ANALYSIS")
    print("=" * 80)
    
    print("\n[1] SPARSIFICATION: Impact of removing high-uncertainty boxes")
    print("-" * 80)
    print(f"{'Threshold':<12} {'Kept':<10} {'Removed':<10} {'mAP':<8} {'Δ mAP':<8} {'Prec':<8} {'Rec':<8}")
    print("-" * 80)
    
    baseline_map = None
    for threshold in sorted(sparsification_results.keys()):
        metrics = sparsification_results[threshold]
        if baseline_map is None:
            baseline_map = metrics.map_at_threshold
        print(
            f"{threshold:<12.3f} {metrics.num_boxes_kept:<10} {metrics.num_boxes_removed:<10} "
            f"{metrics.map_at_threshold:<8.3f} {metrics.map_improvement:<8.3f} "
            f"{metrics.precision:<8.3f} {metrics.recall:<8.3f}"
        )
    print()
    
    print("\n[2] UNCERTAINTY-vs-ERROR CORRELATION")
    print("-" * 80)
    print(f"Spearman correlation: {correlation_results['spearman_corr']:.4f}")
    print(f"Pearson correlation:  {correlation_results['pearson_corr']:.4f}")
    print(f"Matched detections:   {correlation_results['num_matched_detections']}")
    print()
    print(f"{'Unc Bin':<12} {'Mean Err':<12} {'Std Err':<12} {'Count':<10} {'Mean IoU':<10}")
    print("-" * 80)
    for bin_c, err, std, cnt, iou in zip(
        correlation_results["uncertainty_bins"],
        correlation_results["mean_errors"],
        correlation_results["std_errors"],
        correlation_results["count_per_bin"],
        correlation_results["mean_iou_per_bin"]
    ):
        print(f"{bin_c:<12.4f} {err:<12.4f} {std:<12.4f} {cnt:<10} {iou:<10.4f}")
    print()
    
    print("=" * 80)
    print(f"Results saved to {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibration analysis")
    parser.add_argument("--detections-json", required=True,
                       help="JSON file with MC inference results")
    parser.add_argument("--ground-truth-json", required=True,
                       help="JSON file with ground truth")
    parser.add_argument("--output-dir", default="results/calibration",
                       help="Output directory for plots and metrics")
    
    args = parser.parse_args()
    
    with open(args.detections_json) as f:
        detections = json.load(f)
    with open(args.ground_truth_json) as f:
        ground_truth = json.load(f)
    
    sparsif = sparsification_analysis(detections, ground_truth)
    corr = uncertainty_error_correlation(detections, ground_truth)
    
    save_calibration_report(sparsif, corr, args.output_dir)
