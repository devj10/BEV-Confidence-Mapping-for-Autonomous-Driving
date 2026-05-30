#!/usr/bin/env python3
"""
sanity_checks.py

Phase 4 Evaluation: Sanity checks for MC-Dropout uncertainty.

Implements two key comparisons:
  1. Single-pass vs. 20-pass: does MC aggregation improve or maintain baseline mAP?
  2. Standard Dropout vs. DropBlock: does DropBlock provide better uncertainty?

Usage:
    from bev_uncertainty.eval.sanity_checks import compare_single_vs_mc, compare_dropout_vs_dropblock
    
    # Compare 1-pass vs 20-pass aggregated boxes
    comparison = compare_single_vs_mc(
        single_pass_detections,
        mc_aggregated_detections,
        ground_truth_per_frame
    )
    
    # Compare uncertainty quality under different dropout strategies
    db_comp = compare_dropout_vs_dropblock(
        std_dropout_detections,
        dropblock_detections,
        ground_truth_per_frame
    )
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ComparisonMetrics:
    """Metrics for a single detection set."""
    name: str
    num_detections: int
    num_frames: int
    precision: float
    recall: float
    map_score: float
    mean_uncertainty: float
    uncertainty_std: float
    min_uncertainty: float
    max_uncertainty: float


@dataclass
class PairwiseComparison:
    """Side-by-side comparison of two detection sets."""
    baseline: ComparisonMetrics
    test: ComparisonMetrics
    map_delta: float
    precision_delta: float
    recall_delta: float
    mean_unc_ratio: float  # test / baseline


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
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
) -> Tuple[int, int, int]:
    """
    Match predictions to ground truth.
    Returns: (tp, fp, fn)
    """
    matched = np.zeros(len(detections), dtype=bool)
    used_gt = set()
    
    # Sort by confidence (descending)
    sorted_idx = np.argsort([-d.get("conf", d.get("score", 0.5)) for d in detections])
    
    for det_idx in sorted_idx:
        det = detections[det_idx]
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in used_gt:
                continue
            
            if not class_agnostic and det.get("class_id") != gt.get("class_id"):
                continue
            
            iou = compute_iou(np.array(det["xyxy"]), np.array(gt["xyxy"]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched[det_idx] = True
            used_gt.add(best_gt_idx)
    
    tp = np.sum(matched)
    fp = len(detections) - tp
    fn = len(ground_truth) - tp
    
    return tp, fp, fn


def compute_metrics_for_detections(
    detections_per_frame: List[List[Dict[str, Any]]],
    ground_truth_per_frame: List[List[Dict[str, Any]]],
    name: str = "detections",
    iou_thresh: float = 0.5,
    class_agnostic: bool = False
) -> ComparisonMetrics:
    """
    Compute aggregate metrics for a set of detections.
    
    Args:
        detections_per_frame: list of frames, each with list of detections
        ground_truth_per_frame: corresponding ground truth
        name: name for this detection set
        iou_thresh: IoU threshold for matching
        class_agnostic: ignore class when matching
    
    Returns:
        ComparisonMetrics with precision, recall, mAP, and uncertainty stats
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_uncertainties = []
    total_dets = 0
    
    for dets, gts in zip(detections_per_frame, ground_truth_per_frame):
        tp, fp, fn = match_detections_to_ground_truth(
            dets, gts, iou_thresh=iou_thresh, class_agnostic=class_agnostic
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_dets += len(dets)
        
        # Collect uncertainties
        for det in dets:
            unc = det.get("uncertainty", 0.0)
            all_uncertainties.append(unc)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    map_score = precision * recall if recall > 0 else 0.0
    
    if all_uncertainties:
        mean_unc = np.mean(all_uncertainties)
        std_unc = np.std(all_uncertainties)
        min_unc = np.min(all_uncertainties)
        max_unc = np.max(all_uncertainties)
    else:
        mean_unc = std_unc = min_unc = max_unc = 0.0
    
    return ComparisonMetrics(
        name=name,
        num_detections=total_dets,
        num_frames=len(detections_per_frame),
        precision=precision,
        recall=recall,
        map_score=map_score,
        mean_uncertainty=mean_unc,
        uncertainty_std=std_unc,
        min_uncertainty=min_unc,
        max_uncertainty=max_unc
    )


def compare_single_vs_mc(
    single_pass_detections: List[List[Dict[str, Any]]],
    mc_aggregated_detections: List[List[Dict[str, Any]]],
    ground_truth_per_frame: List[List[Dict[str, Any]]],
    iou_thresh: float = 0.5,
    class_agnostic: bool = False
) -> PairwiseComparison:
    """
    Sanity check #1: Compare single-pass baseline vs. multi-pass (T=20) aggregation.
    
    Hypothesis: 
      - MC aggregation should maintain or improve mAP (by suppressing false positives)
      - Single-pass has zero uncertainty; MC pass has non-zero uncertainty
    
    Args:
        single_pass_detections: detections from a single forward pass
        mc_aggregated_detections: detections from T=20 aggregation
        ground_truth_per_frame: ground truth
    
    Returns:
        PairwiseComparison with both sets of metrics and deltas
    """
    baseline = compute_metrics_for_detections(
        single_pass_detections,
        ground_truth_per_frame,
        name="Single-Pass Baseline",
        iou_thresh=iou_thresh,
        class_agnostic=class_agnostic
    )
    
    test = compute_metrics_for_detections(
        mc_aggregated_detections,
        ground_truth_per_frame,
        name="MC Aggregated (T=20)",
        iou_thresh=iou_thresh,
        class_agnostic=class_agnostic
    )
    
    return PairwiseComparison(
        baseline=baseline,
        test=test,
        map_delta=test.map_score - baseline.map_score,
        precision_delta=test.precision - baseline.precision,
        recall_delta=test.recall - baseline.recall,
        mean_unc_ratio=test.mean_uncertainty / (baseline.mean_uncertainty + 1e-9)
    )


def compare_dropout_vs_dropblock(
    std_dropout_detections: List[List[Dict[str, Any]]],
    dropblock_detections: List[List[Dict[str, Any]]],
    ground_truth_per_frame: List[List[Dict[str, Any]]],
    iou_thresh: float = 0.5,
    class_agnostic: bool = False
) -> PairwiseComparison:
    """
    Sanity check #2: Compare standard Dropout vs. DropBlock uncertainty.
    
    Hypothesis:
      - DropBlock provides better spatial uncertainty than point-wise dropout
      - DropBlock uncertainty should correlate better with error
      - DropBlock may achieve similar or better mAP with more meaningful uncertainty
    
    Args:
        std_dropout_detections: detections using standard Dropout
        dropblock_detections: detections using DropBlock
        ground_truth_per_frame: ground truth
    
    Returns:
        PairwiseComparison
    """
    baseline = compute_metrics_for_detections(
        std_dropout_detections,
        ground_truth_per_frame,
        name="Standard Dropout",
        iou_thresh=iou_thresh,
        class_agnostic=class_agnostic
    )
    
    test = compute_metrics_for_detections(
        dropblock_detections,
        ground_truth_per_frame,
        name="DropBlock",
        iou_thresh=iou_thresh,
        class_agnostic=class_agnostic
    )
    
    return PairwiseComparison(
        baseline=baseline,
        test=test,
        map_delta=test.map_score - baseline.map_score,
        precision_delta=test.precision - baseline.precision,
        recall_delta=test.recall - baseline.recall,
        mean_unc_ratio=test.mean_uncertainty / (baseline.mean_uncertainty + 1e-9)
    )


def print_comparison(comparison: PairwiseComparison) -> None:
    """Pretty-print a pairwise comparison."""
    print("\n" + "=" * 100)
    print(f"COMPARISON: {comparison.baseline.name} vs. {comparison.test.name}")
    print("=" * 100)
    
    print(f"\n{'Metric':<30} {comparison.baseline.name:<30} {comparison.test.name:<30} {'Δ':>8}")
    print("-" * 100)
    
    metrics = [
        ("Frames", f"{comparison.baseline.num_frames}", f"{comparison.test.num_frames}", "—"),
        ("Total Detections", f"{comparison.baseline.num_detections}", f"{comparison.test.num_detections}", "—"),
        ("Precision", f"{comparison.baseline.precision:.4f}", f"{comparison.test.precision:.4f}", 
         f"{comparison.precision_delta:+.4f}"),
        ("Recall", f"{comparison.baseline.recall:.4f}", f"{comparison.test.recall:.4f}", 
         f"{comparison.recall_delta:+.4f}"),
        ("mAP", f"{comparison.baseline.map_score:.4f}", f"{comparison.test.map_score:.4f}", 
         f"{comparison.map_delta:+.4f}"),
        ("—", "—", "—", "—"),
        ("Mean Uncertainty", f"{comparison.baseline.mean_uncertainty:.6f}", f"{comparison.test.mean_uncertainty:.6f}", 
         f"{comparison.test.mean_uncertainty - comparison.baseline.mean_uncertainty:+.6f}"),
        ("Std Uncertainty", f"{comparison.baseline.uncertainty_std:.6f}", f"{comparison.test.uncertainty_std:.6f}", 
         f"{comparison.test.uncertainty_std - comparison.baseline.uncertainty_std:+.6f}"),
        ("Min Uncertainty", f"{comparison.baseline.min_uncertainty:.6f}", f"{comparison.test.min_uncertainty:.6f}", 
         "—"),
        ("Max Uncertainty", f"{comparison.baseline.max_uncertainty:.6f}", f"{comparison.test.max_uncertainty:.6f}", 
         "—"),
    ]
    
    for row in metrics:
        if row[0] == "—":
            print("-" * 100)
        else:
            print(f"{row[0]:<30} {row[1]:<30} {row[2]:<30} {row[3]:>8}")
    
    print("=" * 100 + "\n")


def save_comparison_report(
    comparison: PairwiseComparison,
    output_dir: Path,
    name: str = "comparison"
) -> None:
    """Save comparison metrics to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "baseline": asdict(comparison.baseline),
        "test": asdict(comparison.test),
        "deltas": {
            "map": comparison.map_delta,
            "precision": comparison.precision_delta,
            "recall": comparison.recall_delta,
            "mean_unc_ratio": comparison.mean_unc_ratio
        }
    }
    
    path = output_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved comparison to {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sanity checks for MC-Dropout")
    parser.add_argument("--single-pass-json", help="JSON with single-pass detections")
    parser.add_argument("--mc-json", help="JSON with MC-aggregated detections")
    parser.add_argument("--std-dropout-json", help="JSON with standard Dropout detections")
    parser.add_argument("--dropblock-json", help="JSON with DropBlock detections")
    parser.add_argument("--ground-truth-json", required=True, help="JSON with ground truth")
    parser.add_argument("--output-dir", default="results/sanity_checks", help="Output directory")
    
    args = parser.parse_args()
    
    with open(args.ground_truth_json) as f:
        ground_truth = json.load(f)
    
    # Check 1: Single vs MC
    if args.single_pass_json and args.mc_json:
        with open(args.single_pass_json) as f:
            single_pass = json.load(f)
        with open(args.mc_json) as f:
            mc = json.load(f)
        
        comp = compare_single_vs_mc(single_pass, mc, ground_truth)
        print_comparison(comp)
        save_comparison_report(comp, args.output_dir, "single_vs_mc")
    
    # Check 2: Dropout vs DropBlock
    if args.std_dropout_json and args.dropblock_json:
        with open(args.std_dropout_json) as f:
            std_dropout = json.load(f)
        with open(args.dropblock_json) as f:
            dropblock = json.load(f)
        
        comp = compare_dropout_vs_dropblock(std_dropout, dropblock, ground_truth)
        print_comparison(comp)
        save_comparison_report(comp, args.output_dir, "dropout_vs_dropblock")
