#!/usr/bin/env python3
"""
Batch MC inference on all nuScenes mini samples.

Generates MC detections (10 stochastic passes) for all CAM_FRONT images,
aggregates them into a single results file for downstream evaluation.

Usage:
    python scripts/batch_mc_inference.py --model runs/detect/.../weights/best.pt --output-dir results/mc_batch
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from nuscenes.nuscenes import NuScenes
from ultralytics import YOLO


def load_model(model_path: str, device: str = None) -> YOLO:
    """Load YOLOv8 model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    print(f"Using device: {device}")
    model = YOLO(model_path)
    
    if device == "cuda" and torch.cuda.is_available():
        model.to(device)
    
    return model


def run_mc_inference_on_image(
    model: YOLO,
    image_path: str,
    num_passes: int = 10,
    conf_thresh: float = 0.5,
) -> Dict:
    """
    Run MC inference with dropout enabled.
    
    Returns:
        {
            'passes': [
                {'detections': [...], 'conf': [...], 'uncertainty': [...]},
                ...
            ],
            'aggregated': {
                'mean_conf': [...],
                'std_conf': [...],
                'uncertainty': [...]
            }
        }
    """
    # Enable dropout by setting model to train mode
    model.model.train()
    
    passes = []
    all_confs = []
    
    for pass_idx in range(num_passes):
        # Run inference
        results = model.predict(
            image_path,
            conf=conf_thresh,
            save=False,
            verbose=False,
            device="cpu" if not torch.cuda.is_available() else "cuda",
        )
        
        if not results:
            continue
        
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            passes.append({'detections': [], 'conf': []})
            continue
        
        # Extract detections
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        detections = []
        for box, conf, cls_id in zip(boxes, confs, classes):
            detections.append({
                'xyxy': box.tolist(),
                'conf': float(conf),
                'class_id': int(cls_id),
            })
        
        passes.append({
            'detections': detections,
            'conf': confs.tolist(),
        })
        all_confs.append(confs)
    
    # Aggregate across passes
    aggregated = {
        'num_passes': num_passes,
        'num_successful': len(passes),
    }
    
    if all_confs:
        all_confs = np.array(all_confs)
        mean_conf = np.mean(all_confs, axis=0) if all_confs.size > 0 else []
        std_conf = np.std(all_confs, axis=0) if all_confs.size > 0 else []
        aggregated['mean_conf'] = mean_conf.tolist() if len(mean_conf) > 0 else []
        aggregated['std_conf'] = std_conf.tolist() if len(std_conf) > 0 else []
    
    return {
        'passes': passes,
        'aggregated': aggregated,
    }


def batch_mc_inference(
    model_path: str,
    output_dir: str = "results/mc_batch",
    num_passes: int = 10,
    conf_thresh: float = 0.5,
    max_samples: int = None,
) -> None:
    """
    Run MC inference on all samples.
    
    Args:
        model_path: Path to YOLOv8 checkpoint
        output_dir: Output directory for results
        num_passes: Number of MC passes per image
        conf_thresh: Confidence threshold
        max_samples: Limit number of samples (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BATCH MC INFERENCE ON ALL SAMPLES")
    print("=" * 80)
    
    # Load model
    model = load_model(model_path)
    
    # Load dataset
    print("\nLoading nuScenes dataset...")
    nusc = NuScenes(version='v1.0-mini', dataroot='data/v1.0-mini', verbose=False)
    print(f"✓ Dataset loaded ({len(nusc.sample)} samples)")
    
    # Process samples
    num_to_process = min(len(nusc.sample), max_samples) if max_samples else len(nusc.sample)
    print(f"\nProcessing {num_to_process} samples with {num_passes} MC passes each...\n")
    
    all_results = {}
    statistics = {
        'total_samples': num_to_process,
        'successful': 0,
        'failed': 0,
        'detections_per_sample': [],
    }
    
    for sample_idx, sample in enumerate(tqdm(nusc.sample[:num_to_process], desc="Samples")):
        sample_token = sample['token']
        scene_token = sample['scene_token']
        timestamp = sample['timestamp']
        
        try:
            # Get camera data
            cam_token = sample['data']['CAM_FRONT']
            cam_data = nusc.get('sample_data', cam_token)
            image_path = f"data/v1.0-mini/{cam_data['filename']}"
            
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Run MC inference
            mc_result = run_mc_inference_on_image(
                model,
                image_path,
                num_passes=num_passes,
                conf_thresh=conf_thresh,
            )
            
            # Store results
            result = {
                'sample_token': sample_token,
                'scene_token': scene_token,
                'timestamp': timestamp,
                'image_path': cam_data['filename'],
                'mc_passes': mc_result['passes'],
                'aggregated': mc_result['aggregated'],
            }
            
            all_results[sample_token] = result
            
            # Count detections
            total_det = 0
            for pass_data in mc_result['passes']:
                total_det += len(pass_data['detections'])
            statistics['detections_per_sample'].append(total_det // num_passes)
            
            statistics['successful'] += 1
            
        except Exception as e:
            print(f"\n❌ Error processing sample {sample_idx}: {e}")
            statistics['failed'] += 1
    
    # Save aggregated results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    output_json = output_path / "mc_detections_all_samples.json"
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved {statistics['successful']} samples to {output_json}")
    
    # Save statistics
    if statistics['detections_per_sample']:
        statistics['avg_detections'] = float(np.mean(statistics['detections_per_sample']))
        statistics['std_detections'] = float(np.std(statistics['detections_per_sample']))
        statistics['min_detections'] = int(np.min(statistics['detections_per_sample']))
        statistics['max_detections'] = int(np.max(statistics['detections_per_sample']))
    
    stats_json = output_path / "inference_statistics.json"
    with open(stats_json, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"✓ Saved statistics to {stats_json}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)
    print(f"  Total samples:      {statistics['total_samples']}")
    print(f"  Successful:         {statistics['successful']}")
    print(f"  Failed:             {statistics['failed']}")
    
    if statistics['detections_per_sample']:
        print(f"  Avg detections:     {statistics['avg_detections']:.1f} per image")
        print(f"  Range:              {statistics['min_detections']}-{statistics['max_detections']}")
    
    print(f"\nResults saved to: {output_path}/")


def main():
    parser = argparse.ArgumentParser(description="Batch MC inference on all samples")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLOv8 checkpoint"
    )
    parser.add_argument("--output-dir", default="results/mc_batch")
    parser.add_argument("--num-passes", type=int, default=10)
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    
    args = parser.parse_args()
    
    batch_mc_inference(
        model_path=args.model,
        output_dir=args.output_dir,
        num_passes=args.num_passes,
        conf_thresh=args.conf_thresh,
        max_samples=args.max_samples,
    )
    
    print("\n" + "=" * 80)
    print("✅ BATCH INFERENCE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
