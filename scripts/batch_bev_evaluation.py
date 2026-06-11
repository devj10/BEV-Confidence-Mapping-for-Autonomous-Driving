#!/usr/bin/env python3
"""
Batch evaluation on all nuScenes mini samples.

Pipeline for each sample:
  1. MC inference with DropBlock (10 passes)
  2. BEV lifting (2D → 3D)
  3. BEV metrics computation
  4. Results aggregation

Outputs:
  - Per-sample: JSON metrics + visualization
  - Summary: Aggregated statistics across all samples
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from nuscenes.nuscenes import NuScenes
from scripts.validate_bev_lifting import (
    load_mc_detections, get_gt_boxes, lift_detections
)
from eval.bev_metrics import BEVMetrics, print_metrics


def run_batch_evaluation(
    mc_results_file: str = "results/mc_batch/mc_detections_all_samples.json",
    output_dir: str = "results/batch_evaluation",
    distance_thresh: float = 1.0,
) -> Dict:
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BATCH EVALUATION ON ALL SAMPLES")
    print("=" * 80)
    
    # Load MC results
    if not Path(mc_results_file).exists():
        print(f"MC results file not found: {mc_results_file}")
        return {}, {}
    
    print(f"\n[1] Loading MC detections from {mc_results_file}...")
    with open(mc_results_file, 'r') as f:
        mc_all_results = json.load(f)
    print(f"✓ Loaded {len(mc_all_results)} samples with MC detections")
    
    # Load dataset
    print("\n[2] Loading nuScenes dataset...")
    nusc = NuScenes(version='v1.0-mini', dataroot='data/v1.0-mini', verbose=False)
    print(f"✓ Dataset loaded ({len(nusc.sample)} samples)")
    
    # Metrics
    metrics_computer = BEVMetrics(distance_thresh=distance_thresh)
    all_results = []
    sample_metrics = {}
    
    print(f"\n[3] Processing {len(mc_all_results)} samples with MC results...\n")
    
    for sample_token in tqdm(mc_all_results.keys(), desc="Samples"):
        mc_result = mc_all_results[sample_token]
        
        try:
            # Get sample
            sample = nusc.get('sample', sample_token)
            timestamp = sample['timestamp']
            
            # Get camera data
            cam_token = sample['data']['CAM_FRONT']
            cam_data = nusc.get('sample_data', cam_token)
            calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
            
            # Load GT boxes
            gt_boxes = get_gt_boxes(nusc, sample_token, ego_pose)
            
            # Convert MC result to detection format
            detections = []
            for pass_data in mc_result['mc_passes']:
                for det in pass_data['detections']:
                    detections.append({
                        'xyxy': det['xyxy'],
                        'conf': det['conf'],
                        'class_id': det.get('class_id', 0),
                    })
            
            if not detections:
                sample_metrics[sample_token] = {
                    'timestamp': timestamp,
                    'status': 'no_detections',
                }
                continue
            
            # Lift to 3D
            detections_3d = lift_detections(detections, gt_boxes, calib, ego_pose)
            
            # Compute metrics
            metrics = metrics_computer.evaluate(detections_3d, gt_boxes)
            
            # Add metadata
            metrics['timestamp'] = timestamp
            metrics['sample_token'] = sample_token[:8]
            metrics['num_mc_passes'] = mc_result['aggregated']['num_passes']
            
            sample_metrics[sample_token] = metrics
            all_results.append(metrics['overall'])
            
        except Exception as e:
            sample_metrics[sample_token] = {
                'timestamp': timestamp,
                'status': 'error',
                'error': str(e)
            }
    
    print("\n" + "=" * 80)
    print("AGGREGATING RESULTS")
    print("=" * 80)
    
    if all_results:
        recall_values = [r['recall'] for r in all_results]
        precision_values = [r['precision'] for r in all_results]
        f1_values = [r['f1'] for r in all_results]
        error_values = [r['avg_error'] for r in all_results if r['avg_error'] > 0]
        
        aggregated = {
            'total_samples': len(mc_all_results),
            'successful_samples': len(all_results),
            'recall': {
                'mean': float(np.mean(recall_values)),
                'std': float(np.std(recall_values)),
                'min': float(np.min(recall_values)),
                'max': float(np.max(recall_values)),
            },
            'precision': {
                'mean': float(np.mean(precision_values)),
                'std': float(np.std(precision_values)),
                'min': float(np.min(precision_values)),
                'max': float(np.max(precision_values)),
            },
            'f1': {
                'mean': float(np.mean(f1_values)),
                'std': float(np.std(f1_values)),
                'min': float(np.min(f1_values)),
                'max': float(np.max(f1_values)),
            },
            'avg_error': {
                'mean': float(np.mean(error_values)) if error_values else 0.0,
                'std': float(np.std(error_values)) if error_values else 0.0,
                'min': float(np.min(error_values)) if error_values else 0.0,
                'max': float(np.max(error_values)) if error_values else 0.0,
            } if error_values else None,
        }
    else:
        aggregated = {
            'total_samples': len(mc_all_results),
            'successful_samples': 0,
            'note': 'No detections available - run MC inference first'
        }
    
    # Print summary
    print("\n[SUMMARY STATISTICS]")
    print(f"  Total samples:          {aggregated['total_samples']}")
    print(f"  Successful:             {aggregated['successful_samples']}")
    
    if aggregated['successful_samples'] > 0:
        print(f"\n  Recall (mean ± std):    {aggregated['recall']['mean']:.3f} ± {aggregated['recall']['std']:.3f}")
        print(f"  Precision (mean ± std): {aggregated['precision']['mean']:.3f} ± {aggregated['precision']['std']:.3f}")
        print(f"  F1-Score (mean ± std):  {aggregated['f1']['mean']:.3f} ± {aggregated['f1']['std']:.3f}")
        
        if aggregated['avg_error']:
            print(f"  Avg Error (mean ± std): {aggregated['avg_error']['mean']:.3f} ± {aggregated['avg_error']['std']:.3f} m")
    
    # Save results
    print("\n[SAVING RESULTS]")
    
    results_json = output_path / "aggregated_metrics.json"
    with open(results_json, 'w') as f:
        json.dump({
            'aggregated': aggregated,
            'per_sample': sample_metrics,
        }, f, indent=2)
    print(f"✓ Saved to {results_json}")
    
    per_sample_json = output_path / "per_sample_metrics.json"
    with open(per_sample_json, 'w') as f:
        json.dump(sample_metrics, f, indent=2)
    print(f"✓ Saved to {per_sample_json}")
    
    return aggregated, sample_metrics


def print_sample_ranking(sample_metrics: Dict):
    """Print samples ranked by F1-score."""
    print("\n" + "=" * 80)
    print("SAMPLE RANKING (by F1-score)")
    print("=" * 80)
    
    ranked = []
    for token, metrics in sample_metrics.items():
        if 'overall' in metrics:
            f1 = metrics['overall']['f1']
            recall = metrics['overall']['recall']
            precision = metrics['overall']['precision']
            ranked.append({
                'token': token[:8],
                'f1': f1,
                'recall': recall,
                'precision': precision,
                'timestamp': metrics.get('timestamp', 'N/A')
            })
    
    ranked.sort(key=lambda x: x['f1'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Token':<10} {'F1':<8} {'Recall':<10} {'Precision':<12} {'Timestamp':<20}")
    print("-" * 80)
    
    for i, sample in enumerate(ranked[:20], 1):  # Show top 20
        print(
            f"{i:<6} {sample['token']:<10} {sample['f1']:<8.3f} "
            f"{sample['recall']:<10.3f} {sample['precision']:<12.3f} {str(sample['timestamp']):<20}"
        )


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation on all samples")
    parser.add_argument(
        "--mc-results",
        default="results/mc_batch/mc_detections_all_samples.json",
        help="Path to MC inference results JSON"
    )
    parser.add_argument("--output-dir", default="results/batch_evaluation")
    parser.add_argument("--distance-thresh", type=float, default=1.0)
    args = parser.parse_args()
    
    aggregated, sample_metrics = run_batch_evaluation(
        mc_results_file=args.mc_results,
        output_dir=args.output_dir,
        distance_thresh=args.distance_thresh,
    )
    
    if sample_metrics:
        print_sample_ranking(sample_metrics)
    
    print("\n" + "=" * 80)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  • aggregated_metrics.json")
    print(f"  • per_sample_metrics.json")


if __name__ == "__main__":
    main()
