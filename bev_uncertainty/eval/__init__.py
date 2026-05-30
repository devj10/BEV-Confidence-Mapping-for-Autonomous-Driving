#!/usr/bin/env python3
"""
bev_uncertainty/eval/__init__.py

Evaluation and calibration modules for Phase 4.
"""

from bev_uncertainty.eval.calibration import (
    sparsification_analysis,
    uncertainty_error_correlation,
    save_calibration_report,
    compute_iou,
    match_detections_to_ground_truth,
    CalibrationMetrics
)

from bev_uncertainty.eval.sanity_checks import (
    compare_single_vs_mc,
    compare_dropout_vs_dropblock,
    compute_metrics_for_detections,
    print_comparison,
    save_comparison_report,
    ComparisonMetrics,
    PairwiseComparison
)

__all__ = [
    # Calibration
    "sparsification_analysis",
    "uncertainty_error_correlation",
    "save_calibration_report",
    "compute_iou",
    "match_detections_to_ground_truth",
    "CalibrationMetrics",
    
    # Sanity checks
    "compare_single_vs_mc",
    "compare_dropout_vs_dropblock",
    "compute_metrics_for_detections",
    "print_comparison",
    "save_comparison_report",
    "ComparisonMetrics",
    "PairwiseComparison"
]
