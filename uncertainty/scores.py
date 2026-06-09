#!/usr/bin/env python3
"""
scores.py

Uncertainty score definitions for MC-Dropout detections.

Implements multiple uncertainty metrics:
  1. Center variance — spatial spread of box centers across T passes
  2. Box variance — variation in box coordinates (xyxy)
  3. Entropy — prediction entropy (for soft class scores)
  4. Confidence variance — variation in detection confidence across passes
  5. Combined score — weighted ensemble of above metrics

Usage:
    from uncertainty.scores import compute_center_variance, compute_box_variance, compute_combined_score
    
    # After aggregation, compute uncertainty scores
    uncertainty = compute_center_variance(clustered_boxes)
    
    # Or use a composite score
    score = compute_combined_score(
        center_var=center_var,
        box_var=box_var,
        conf_var=conf_var,
        weights={"center": 0.5, "box": 0.3, "conf": 0.2}
    )
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


# ────────────────────────────────────────────────────────────────────────────
# 1. CENTER VARIANCE — Spatial spread of box centers
# ────────────────────────────────────────────────────────────────────────────

def compute_box_center(xyxy: List[float]) -> Tuple[float, float]:
    """
    Compute center point of a box.
    
    Args:
        xyxy: [x1, y1, x2, y2]
    
    Returns:
        (center_x, center_y)
    """
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def compute_center_variance(
    boxes: List[List[float]],
    normalize: bool = False,
    image_size: int = 640
) -> float:
    """
    Compute variance of box centers across multiple predictions.
    
    This is the primary uncertainty metric used in Phase 3 aggregation.
    High variance → box centers are scattered → high uncertainty
    Low variance → box centers are consistent → low uncertainty
    
    Args:
        boxes: list of [x1, y1, x2, y2] in pixel coordinates
        normalize: if True, normalize by image size
        image_size: image width/height (for normalization)
    
    Returns:
        Scalar variance value (0 = certain, >0 = uncertain)
    
    Example:
        boxes = [
            [100, 100, 200, 200],  # Pass 1
            [101, 102, 201, 202],  # Pass 2
            [99, 98, 199, 198]     # Pass 3
        ]
        unc = compute_center_variance(boxes)  # ~1-2 pixels
    """
    if len(boxes) < 2:
        return 0.0
    
    centers = np.array([compute_box_center(box) for box in boxes])
    
    # Compute variance in each dimension, then combine
    var_x = np.var(centers[:, 0])
    var_y = np.var(centers[:, 1])
    variance = (var_x + var_y) / 2.0
    
    if normalize:
        # Normalize by image area to make scale-invariant
        variance = variance / (image_size ** 2)
    
    return float(variance)


# ────────────────────────────────────────────────────────────────────────────
# 2. BOX VARIANCE — Variation in box dimensions
# ────────────────────────────────────────────────────────────────────────────

def compute_box_dimensions(xyxy: List[float]) -> Tuple[float, float]:
    """
    Compute width and height of a box.
    
    Args:
        xyxy: [x1, y1, x2, y2]
    
    Returns:
        (width, height)
    """
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    return w, h


def compute_box_variance(
    boxes: List[List[float]],
    normalize: bool = False,
    image_size: int = 640
) -> float:
    """
    Compute variance of box dimensions (width, height) across predictions.
    
    Higher variance → box size is uncertain → high uncertainty
    Lower variance → box size is consistent → low uncertainty
    
    Args:
        boxes: list of [x1, y1, x2, y2] in pixel coordinates
        normalize: if True, normalize by image size
        image_size: image width/height (for normalization)
    
    Returns:
        Scalar variance value
    
    Example:
        boxes = [
            [100, 100, 200, 200],  # 100×100
            [100, 100, 205, 205],  # 105×105 (slightly larger)
            [100, 100, 195, 195]   # 95×95 (slightly smaller)
        ]
        unc = compute_box_variance(boxes)  # ~25-30 (pixel² variance)
    """
    if len(boxes) < 2:
        return 0.0
    
    dims = np.array([compute_box_dimensions(box) for box in boxes])
    
    # Variance of width and height
    var_w = np.var(dims[:, 0])
    var_h = np.var(dims[:, 1])
    variance = (var_w + var_h) / 2.0
    
    if normalize:
        variance = variance / (image_size ** 2)
    
    return float(variance)


# ────────────────────────────────────────────────────────────────────────────
# 3. COORDINATE VARIANCE — Variance in individual coordinates
# ────────────────────────────────────────────────────────────────────────────

def compute_coordinate_variance(
    boxes: List[List[float]],
    normalize: bool = False,
    image_size: int = 640
) -> float:
    """
    Compute variance across all four coordinates [x1, y1, x2, y2].
    
    This is more granular than center variance, capturing uncertainty
    in all corners of the box.
    
    Args:
        boxes: list of [x1, y1, x2, y2]
        normalize: if True, normalize by image size
        image_size: image width/height
    
    Returns:
        Scalar variance value
    """
    if len(boxes) < 2:
        return 0.0
    
    boxes_array = np.array(boxes)
    variance = np.mean(np.var(boxes_array, axis=0))
    
    if normalize:
        variance = variance / (image_size ** 2)
    
    return float(variance)


# ────────────────────────────────────────────────────────────────────────────
# 4. CONFIDENCE VARIANCE — Variation in detection confidence
# ────────────────────────────────────────────────────────────────────────────

def compute_confidence_variance(confidences: List[float]) -> float:
    """
    Compute variance of detection confidence scores across T passes.
    
    High variance → model is uncertain about detectability
    Low variance → model is confident about detection
    
    Args:
        confidences: list of confidence scores [0, 1]
    
    Returns:
        Scalar variance (0–0.25, max at 0.5 confidence)
    
    Example:
        confs = [0.95, 0.92, 0.94]  # Consistent
        → variance ≈ 0.0002
        
        confs = [0.9, 0.7, 0.5]     # Inconsistent
        → variance ≈ 0.027
    """
    if len(confidences) < 2:
        return 0.0
    
    confidences = np.array(confidences)
    variance = float(np.var(confidences))
    
    return variance


# ────────────────────────────────────────────────────────────────────────────
# 5. ENTROPY — Prediction entropy for soft class scores
# ────────────────────────────────────────────────────────────────────────────

def compute_class_entropy(
    class_predictions: List[Dict[int, float]]
) -> float:
    """
    Compute entropy of class predictions across T passes.
    
    For each class, we have T predictions (probability that this box is that class).
    If predictions are consistent across passes, entropy is low.
    If inconsistent, entropy is high.
    
    Args:
        class_predictions: list of {class_id: confidence} dicts, one per pass
    
    Returns:
        Scalar entropy value (higher = more uncertain about class)
    
    Example:
        pass1: {0: 0.95, 1: 0.05}   # Very sure it's class 0
        pass2: {0: 0.94, 1: 0.06}
        pass3: {0: 0.96, 1: 0.04}
        → low entropy (consistent prediction of class 0)
        
        pass1: {0: 0.60, 1: 0.40}   # Ambiguous
        pass2: {0: 0.45, 1: 0.55}
        pass3: {0: 0.50, 1: 0.50}
        → high entropy (confused about class)
    """
    if not class_predictions:
        return 0.0
    
    # Aggregate predictions across passes
    # For each class, compute mean probability across passes
    all_classes = set()
    for pred in class_predictions:
        all_classes.update(pred.keys())
    
    class_probs = {}
    for class_id in all_classes:
        probs = [pred.get(class_id, 0.0) for pred in class_predictions]
        class_probs[class_id] = np.mean(probs)
    
    # Compute entropy of this distribution
    probs = np.array(list(class_probs.values()))
    probs = probs / np.sum(probs)  # Normalize
    
    # Shannon entropy: -Σ p_i * log(p_i)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Normalize to [0, 1] by dividing by max entropy (uniform distribution)
    max_entropy = np.log(len(all_classes))
    if max_entropy > 0:
        entropy = entropy / max_entropy
    
    return float(entropy)


# ────────────────────────────────────────────────────────────────────────────
# 6. COMBINED SCORE — Weighted ensemble of metrics
# ────────────────────────────────────────────────────────────────────────────

def compute_combined_score(
    center_var: Optional[float] = None,
    box_var: Optional[float] = None,
    conf_var: Optional[float] = None,
    class_entropy: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    normalize: bool = True
) -> float:
    """
    Compute a weighted combination of uncertainty metrics.
    
    Allows flexible combination of different uncertainty sources.
    By default, weights are tuned empirically:
      - Center variance: 50% (most important spatially)
      - Box variance: 30% (size uncertainty)
      - Confidence variance: 15% (detectability)
      - Class entropy: 5% (class ambiguity)
    
    Args:
        center_var: spatial spread of box centers (from compute_center_variance)
        box_var: variation in box dimensions (from compute_box_variance)
        conf_var: variation in confidence (from compute_confidence_variance)
        class_entropy: class prediction entropy (from compute_class_entropy)
        weights: dict with keys "center", "box", "conf", "entropy"
                 values sum to 1.0
        normalize: if True, normalize all metrics to [0, 1] range before combining
    
    Returns:
        Scalar uncertainty score (0 = certain, higher = uncertain)
    
    Example:
        score = compute_combined_score(
            center_var=1.5,
            box_var=0.8,
            conf_var=0.02,
            class_entropy=0.3,
            weights={"center": 0.5, "box": 0.3, "conf": 0.15, "entropy": 0.05}
        )
    """
    if weights is None:
        weights = {
            "center": 0.5,
            "box": 0.3,
            "conf": 0.15,
            "entropy": 0.05
        }
    
    # Ensure weights sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    components = {}
    total_score = 0.0
    
    # Center variance (normalized by typical image size)
    if center_var is not None:
        if normalize:
            # Normalize to [0, 1]: assume 0-50 pixels² is normal range
            normalized_var = min(1.0, center_var / 50.0)
        else:
            normalized_var = center_var
        components["center"] = normalized_var
        total_score += weights.get("center", 0) * normalized_var
    
    # Box variance (normalized)
    if box_var is not None:
        if normalize:
            # Normalize to [0, 1]: assume 0-500 pixels² is normal
            normalized_var = min(1.0, box_var / 500.0)
        else:
            normalized_var = box_var
        components["box"] = normalized_var
        total_score += weights.get("box", 0) * normalized_var
    
    # Confidence variance (already in [0, 0.25])
    if conf_var is not None:
        if normalize:
            normalized_var = min(1.0, conf_var * 4.0)  # Scale [0, 0.25] to [0, 1]
        else:
            normalized_var = conf_var
        components["conf"] = normalized_var
        total_score += weights.get("conf", 0) * normalized_var
    
    # Class entropy (already in [0, 1])
    if class_entropy is not None:
        components["entropy"] = class_entropy
        total_score += weights.get("entropy", 0) * class_entropy
    
    return float(total_score)


# ────────────────────────────────────────────────────────────────────────────
# 7. CALIBRATED SCORE — Empirically calibrated combined score
# ────────────────────────────────────────────────────────────────────────────

def compute_calibrated_uncertainty(
    boxes: List[List[float]],
    confidences: List[float],
    class_predictions: Optional[List[Dict[int, float]]] = None,
    image_size: int = 640
) -> float:
    """
    All-in-one function to compute uncertainty from MC predictions.
    
    This is the recommended way to compute uncertainty for a clustered detection.
    Internally computes all metrics and combines them with empirically-tuned weights.
    
    Args:
        boxes: list of [x1, y1, x2, y2] from T passes
        confidences: list of detection confidences from T passes
        class_predictions: optional, list of {class_id: conf} from each pass
        image_size: image width/height for normalization
    
    Returns:
        Scalar uncertainty (0–1, where 1 = maximum uncertainty)
    
    Example:
        # After clustering T=20 passes for one object:
        boxes_for_object = [
            [100, 100, 200, 200],  # Pass 1
            [101, 102, 201, 202],  # Pass 2
            ...
        ]
        confidences_for_object = [0.95, 0.92, ..., 0.94]
        
        uncertainty = compute_calibrated_uncertainty(
            boxes_for_object,
            confidences_for_object
        )
    """
    center_var = compute_center_variance(boxes, normalize=False, image_size=image_size)
    box_var = compute_box_variance(boxes, normalize=False, image_size=image_size)
    conf_var = compute_confidence_variance(confidences)
    entropy = compute_class_entropy(class_predictions) if class_predictions else 0.0
    
    return compute_combined_score(
        center_var=center_var,
        box_var=box_var,
        conf_var=conf_var,
        class_entropy=entropy,
        normalize=True
    )


# ────────────────────────────────────────────────────────────────────────────
# 8. ANALYSIS & DEBUGGING
# ────────────────────────────────────────────────────────────────────────────

def analyze_uncertainty_components(
    boxes: List[List[float]],
    confidences: List[float],
    class_predictions: Optional[List[Dict[int, float]]] = None,
    image_size: int = 640,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Decompose uncertainty into components for debugging/analysis.
    
    Returns all individual components so you can see which contributes most.
    
    Args:
        boxes: list of [x1, y1, x2, y2]
        confidences: list of confidences
        class_predictions: optional class predictions
        image_size: for normalization
        verbose: if True, print detailed breakdown
    
    Returns:
        Dict with all computed metrics
    
    Example:
        breakdown = analyze_uncertainty_components(boxes, confidences)
        print(f"Center variance: {breakdown['center_var']:.6f}")
        print(f"Box variance:    {breakdown['box_var']:.6f}")
        print(f"Conf variance:   {breakdown['conf_var']:.6f}")
        print(f"Combined score:  {breakdown['combined']:.6f}")
    """
    center_var = compute_center_variance(boxes, normalize=False, image_size=image_size)
    box_var = compute_box_variance(boxes, normalize=False, image_size=image_size)
    conf_var = compute_confidence_variance(confidences)
    entropy = compute_class_entropy(class_predictions) if class_predictions else 0.0
    coord_var = compute_coordinate_variance(boxes, normalize=False, image_size=image_size)
    
    combined = compute_combined_score(
        center_var=center_var,
        box_var=box_var,
        conf_var=conf_var,
        class_entropy=entropy,
        normalize=True
    )
    
    results = {
        "center_var": center_var,
        "box_var": box_var,
        "coord_var": coord_var,
        "conf_var": conf_var,
        "class_entropy": entropy,
        "combined": combined,
        "num_passes": len(boxes),
        "mean_confidence": float(np.mean(confidences))
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("UNCERTAINTY COMPONENT ANALYSIS")
        print("=" * 70)
        print(f"Center variance:       {center_var:12.6f}  (spatial spread)")
        print(f"Box variance:          {box_var:12.6f}  (size variation)")
        print(f"Coordinate variance:   {coord_var:12.6f}  (all coords)")
        print(f"Confidence variance:   {conf_var:12.6f}  (detection confidence)")
        print(f"Class entropy:         {entropy:12.6f}  (class ambiguity)")
        print(f"Mean confidence:       {np.mean(confidences):12.6f}")
        print("-" * 70)
        print(f"COMBINED UNCERTAINTY:  {combined:12.6f}")
        print("=" * 70 + "\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
    
    print(__doc__)
    
    # Synthetic example
    print("\n" + "=" * 70)
    print("EXAMPLE: Uncertainty scoring for a single detection")
    print("=" * 70)
    
    # Tight clustering (low uncertainty)
    tight_boxes = [
        [100, 100, 200, 200],
        [100.5, 100.5, 200.5, 200.5],
        [99.5, 99.5, 199.5, 199.5]
    ]
    tight_confs = [0.95, 0.94, 0.96]
    
    print("\n[1] Tight clustering (expected low uncertainty):")
    tight_unc = analyze_uncertainty_components(
        tight_boxes, tight_confs, verbose=True
    )
    
    # Loose clustering (high uncertainty)
    loose_boxes = [
        [100, 100, 200, 200],
        [110, 110, 210, 210],
        [90, 90, 190, 190]
    ]
    loose_confs = [0.95, 0.70, 0.80]
    
    print("\n[2] Loose clustering (expected high uncertainty):")
    loose_unc = analyze_uncertainty_components(
        loose_boxes, loose_confs, verbose=True
    )
    
    print(f"\nUncertainty ratio (loose / tight): {loose_unc['combined'] / (tight_unc['combined'] + 1e-8):.1f}x")
