#!/usr/bin/env python3
"""
draw_uncertainty.py

Visualize detections with uncertainty-based coloring:
  - Green: low uncertainty (confident localization)
  - Red: high uncertainty (scattered detections)

This is one of the most convincing early milestone artifacts—seeing high
uncertainty spike in adverse weather or challenging scenarios.

Usage:
    from viz.draw_uncertainty import draw_detections_on_images
    draw_detections_on_images(
        image_path="path/to/image.jpg",
        detections=[{"xyxy": [...], "uncertainty": 0.05, "class_name": "car"}, ...],
        output_dir="results/viz",
        filename_out="result.jpg",
    )
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


def draw_detections_on_images(
    image_path: str,
    detections: List[Dict[str, Any]],
    output_dir: str = "results/viz",
    filename_out: Optional[str] = None,
) -> str:
    """
    Draw detections with uncertainty-based color on an image.
    
    Args:
        image_path: Path to input image
        detections: List of dicts with keys:
                    - xyxy: [x1, y1, x2, y2]
                    - class_name: str
                    - uncertainty: float (normalized, e.g., 0-0.1)
                    - conf: float (detection confidence)
        output_dir: Directory to save visualization
        filename_out: Output filename (default: <image_name>_uncertainty.jpg)
    
    Returns:
        Path to saved visualization
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]

    # Normalize uncertainties to [0, 1] for coloring
    if len(detections) == 0:
        unc_values = []
    else:
        unc_values = [det.get("uncertainty", 0.0) for det in detections]
        unc_min = min(unc_values) if unc_values else 0
        unc_max = max(unc_values) if unc_values else 1
        # Avoid division by zero
        if unc_max == unc_min:
            unc_max = unc_min + 1e-6

    # Draw each detection
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["xyxy"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Clamp to image bounds
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        # Uncertainty → color (green to red)
        uncertainty = det.get("uncertainty", 0.0)
        if unc_max > unc_min:
            unc_norm = (uncertainty - unc_min) / (unc_max - unc_min)
        else:
            unc_norm = 0.5
        unc_norm = max(0.0, min(1.0, unc_norm))

        # HSV: Hue 120 (green) → 0 (red), keep saturation/value constant
        # In OpenCV, hue is 0-180, so green is ~60, red is 0
        hue = int(120 * (1 - unc_norm))
        color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        color_bgr = tuple(map(int, color_bgr))

        # Draw bounding box
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness)

        # Draw label with class name, confidence, and uncertainty
        class_name = det.get("class_name", "unknown")
        conf = det.get("conf", 0.0)
        label = f"{class_name} (conf={conf:.2f}, unc={uncertainty:.4f})"

        # Put text above the box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # white
        font_thickness = 1

        # Get text size for background
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = x1
        text_y = max(30, y1 - 5)

        # Draw semi-transparent background for text
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            color_bgr,
            -1,  # filled
        )
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Draw text
        cv2.putText(img, label, (text_x, text_y), font, font_scale, font_color, font_thickness)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    if filename_out is None:
        input_name = Path(image_path).stem
        filename_out = f"{input_name}_uncertainty.jpg"

    output_path = output_dir / filename_out
    cv2.imwrite(str(output_path), img)

    return str(output_path)


def draw_comparison(
    image_path: str,
    detections_before: List[Dict[str, Any]],
    detections_after: List[Dict[str, Any]],
    output_dir: str = "results/viz",
    filename_out: Optional[str] = None,
) -> str:
    """
    Side-by-side visualization: before and after uncertainty filtering.
    
    Useful for showing sparsification: remove high-uncertainty boxes,
    quality improves.
    
    Args:
        image_path: Input image
        detections_before: Unfiltered detections
        detections_after: Filtered detections (e.g., unc < threshold)
        output_dir: Output directory
        filename_out: Output filename
    
    Returns:
        Path to saved image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]

    # Create canvas: original image + before + after
    canvas_w = img_w * 3
    canvas_h = img_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Paste original
    canvas[:, :img_w] = img

    # Draw before
    before_img = img.copy()
    _draw_boxes(before_img, detections_before)
    canvas[:, img_w:2*img_w] = before_img

    # Draw after
    after_img = img.copy()
    _draw_boxes(after_img, detections_after)
    canvas[:, 2*img_w:] = after_img

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename_out is None:
        input_name = Path(image_path).stem
        filename_out = f"{input_name}_comparison.jpg"

    output_path = output_dir / filename_out
    cv2.imwrite(str(output_path), canvas)

    return str(output_path)


def _draw_boxes(
    img: np.ndarray,
    detections: List[Dict[str, Any]],
) -> None:
    """
    Helper: draw boxes directly on an image array (in-place).
    """
    img_h, img_w = img.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        uncertainty = det.get("uncertainty", 0.0)
        hue = int(120 * (1 - uncertainty))
        color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        color_bgr = tuple(map(int, color_bgr))

        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
