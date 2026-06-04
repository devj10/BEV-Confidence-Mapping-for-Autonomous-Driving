#!/usr/bin/env python3
"""
Compute BEV metrics for MC detection results.

Pipeline:
  1. Load MC detections (lifted 3D positions)
  2. Load GT boxes (transformed to ego frame)
  3. Match detections to GT using distance-based matching
  4. Compute metrics: recall, precision, F1, error by range bucket
  5. Save results to JSON
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from nuscenes.nuscenes import NuScenes
from eval.bev_metrics import BEVMetrics, print_metrics
from scripts.validate_bev_lifting import (
    load_mc_detections, get_gt_boxes, lift_detections
)


def main():
    parser = argparse.ArgumentParser(description="Compute BEV detection metrics")
    parser.add_argument("--sample-token", default=None)
    parser.add_argument("--detections-json", default="results/mc_finetuned/mc_detections.json")
    parser.add_argument("--depth-mode", default="lidar", choices=["gt", "lidar"])
    parser.add_argument("--distance-thresh", type=float, default=2.0,
                        help="Matching distance threshold in metres")
    parser.add_argument("--output-json", default="results/bev_validation/bev_metrics.json")
    args = parser.parse_args()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"BEV METRICS  [depth-mode={args.depth_mode}  thresh={args.distance_thresh} m]")
    print("=" * 80)

    print("\n[1] Loading nuScenes dataset...")
    nusc = NuScenes(version='v1.0-mini', dataroot='data/v1.0-mini', verbose=False)
    print("✓ Dataset loaded")

    sample_token = args.sample_token or nusc.sample[0]['token']
    sample = nusc.get('sample', sample_token)
    print(f"✓ Sample: {sample_token[:8]}...  ts={sample['timestamp']}")

    print("\n[2] Loading camera calibration...")
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    print("✓ Calibration loaded")

    print("\n[3] Loading GT boxes...")
    gt_boxes = get_gt_boxes(nusc, sample_token, ego_pose)
    print(f"✓ {len(gt_boxes)} GT boxes")

    print("\n[4] Loading MC detections...")
    if not Path(args.detections_json).exists():
        print(f"✗ Not found: {args.detections_json}")
        sys.exit(1)
    detections = load_mc_detections(args.detections_json)
    print(f"✓ {len(detections)} detections")

    print(f"\n[5] Lifting detections ({args.depth_mode} depth)...")
    detections_3d = lift_detections(
        detections, gt_boxes, calib, ego_pose,
        nusc=nusc, sample_token=sample_token, depth_mode=args.depth_mode,
    )
    successful = sum(1 for *_, ok in detections_3d if ok)
    print(f"✓ Lifted {successful}/{len(detections_3d)}")

    print("\n[6] Computing BEV metrics...")
    metrics_computer = BEVMetrics(distance_thresh=args.distance_thresh)
    metrics = metrics_computer.evaluate(detections_3d, gt_boxes)
    print_metrics(metrics)

    print("\n[7] Saving results...")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved → {output_path}")

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\n  Recall:    {metrics['overall']['recall']:.1%}")
    print(f"  Precision: {metrics['overall']['precision']:.1%}")
    print(f"  F1-Score:  {metrics['overall']['f1']:.3f}")
    print(f"  Avg Error: {metrics['overall']['avg_error']:.2f} m")


if __name__ == "__main__":
    main()
