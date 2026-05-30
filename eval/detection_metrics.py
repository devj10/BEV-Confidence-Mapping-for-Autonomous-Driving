#!/usr/bin/env python3
"""
detection_metrics.py

Evaluate a trained YOLOv8 checkpoint on the nuScenes val split and report
per-class and overall mAP@50 / mAP@50-95.

Metrics printed to stdout and optionally saved as JSON for downstream use.

Usage:
    python eval/detection_metrics.py \
        --weights runs/baseline/nuscenes_clear/weights/best.pt \
        --data    /path/to/yolo_out/dataset.yaml \
        [--imgsz  640] \
        [--batch  16] \
        [--conf   0.001] \
        [--iou    0.6] \
        [--save-json results/baseline_metrics.json]
"""

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mAP for a YOLOv8 checkpoint on nuScenes val"
    )
    parser.add_argument("--weights",   required=True,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--data",      required=True,
                        help="dataset.yaml used during training")
    parser.add_argument("--imgsz",     type=int,   default=640)
    parser.add_argument("--batch",     type=int,   default=16)
    parser.add_argument("--conf",      type=float, default=0.001,
                        help="Confidence threshold for val (default: 0.001 — keeps recall high)")
    parser.add_argument("--iou",       type=float, default=0.6,
                        help="IoU threshold for NMS (default: 0.6)")
    parser.add_argument("--device",    default=None,
                        help="Device string: '0', 'cpu', etc. (default: auto)")
    parser.add_argument("--save-json", default=None, metavar="PATH",
                        help="Write metrics dict to this JSON file")
    return parser.parse_args()


# ── Formatting helpers ────────────────────────────────────────────────────────

def print_results(results, class_names: list[str]) -> dict:
    """Pretty-print per-class and overall metrics; return as a plain dict."""
    box = results.box  # ultralytics DetMetrics

    # Overall
    mp   = float(box.mp)    # mean precision
    mr   = float(box.mr)    # mean recall
    map50    = float(box.map50)
    map5095  = float(box.map)

    print("\n" + "=" * 62)
    print(f"  {'Class':<20s}  {'P':>6}  {'R':>6}  {'mAP@50':>8}  {'mAP@50-95':>10}")
    print("-" * 62)

    per_class: dict[str, dict] = {}
    ap50_per  = box.ap50     # (nc,) array — aligned to ap_class_index

    for idx, cls_idx in enumerate(box.ap_class_index):
        name  = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        ap50_c = float(ap50_per[idx])
        ap_c   = float(box.ap[idx])
        # per-class P and R aren't stored separately; use overall as placeholder
        print(f"  {name:<20s}  {'—':>6}  {'—':>6}  {ap50_c:>8.3f}  {ap_c:>10.3f}")
        per_class[name] = {"ap50": ap50_c, "ap50_95": ap_c}

    print("-" * 62)
    print(f"  {'all':<20s}  {mp:>6.3f}  {mr:>6.3f}  {map50:>8.3f}  {map5095:>10.3f}")
    print("=" * 62 + "\n")

    return {
        "overall": {
            "precision": mp,
            "recall":    mr,
            "mAP50":     map50,
            "mAP50_95":  map5095,
        },
        "per_class": per_class,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: checkpoint not found: {weights}", file=sys.stderr)
        print("Train a model first with scripts/train_baseline.py", file=sys.stderr)
        sys.exit(1)

    data = Path(args.data)
    if not data.exists():
        print(f"ERROR: dataset.yaml not found: {data}", file=sys.stderr)
        sys.exit(1)

    print(f"Weights : {weights}")
    print(f"Data    : {data}")
    print(f"Imgsz   : {args.imgsz}  |  Batch: {args.batch}")
    print(f"Conf    : {args.conf}   |  IoU:   {args.iou}")

    model = YOLO(str(weights))

    val_kwargs = dict(
        data    = str(data),
        imgsz   = args.imgsz,
        batch   = args.batch,
        conf    = args.conf,
        iou     = args.iou,
        plots   = True,   # saves confusion matrix + PR curves under the run dir
        save_json = False,
        verbose = False,
    )
    if args.device is not None:
        val_kwargs["device"] = args.device

    results = model.val(**val_kwargs)

    # Class names from the model (loaded from dataset.yaml at training time)
    class_names = list(model.names.values()) if model.names else []

    metrics = print_results(results, class_names)

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"Metrics saved to {out}")


if __name__ == "__main__":
    main()
