#!/usr/bin/env python3

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))

from dropblock import DropBlock2D
from uncertainty.associate import associate_detections
from uncertainty.aggregate import aggregate_clusters
from uncertainty.scores import compute_calibrated_uncertainty
from viz.draw_uncertainty import draw_detections_on_images


class YOLOLayerWithDropBlock(torch.nn.Module):
    def __init__(self, layer, db):
        super().__init__()
        self.layer = layer
        self.db = db

        # Preserve Ultralytics routing metadata
        self.f = layer.f
        self.i = layer.i
        self.type = layer.type
        self.np = getattr(layer, "np", 0)

    def forward(self, x):
        return self.db(self.layer(x))


def inject_dropblock_into_model(model, block_size=3, drop_prob=0.03):
    print(f"\n[MC-DropBlock] Injecting DropBlock (block_size={block_size}, drop_prob={drop_prob})")

    model_net = model.model
    injected_count = 0

    # Safer later YOLO blocks; avoid early backbone convs
    safe_layer_indices = [12, 15, 18]

    for idx in safe_layer_indices:
        if idx >= len(model_net.model):
            continue

        old_layer = model_net.model[idx]
        db = DropBlock2D(block_size=block_size, drop_prob=drop_prob)
        model_net.model[idx] = YOLOLayerWithDropBlock(old_layer, db)
        injected_count += 1

    print(f"✓ Injected DropBlock into {injected_count} YOLO blocks")
    return model


def enable_mc_inference(model, enable=True):
    for module in model.model.modules():
        if isinstance(module, DropBlock2D):
            module.mc_inference = enable

    if enable:
        print("[MC-DropBlock] MC inference mode ENABLED")
    else:
        print("[MC-DropBlock] MC inference mode DISABLED")


def get_class_name(class_id, class_names):
    class_id = int(class_id)

    if isinstance(class_names, dict):
        return class_names.get(class_id, f"class_{class_id}")

    if isinstance(class_names, list):
        if 0 <= class_id < len(class_names):
            return class_names[class_id]

    return f"class_{class_id}"


def run_mc_passes(
    model,
    image_path,
    class_names,
    num_passes=20,
    conf_thresh=0.3,
    iou_thresh=0.5,
):
    print(f"\n[MC Inference] Running {num_passes} stochastic forward passes...")

    all_passes = []
    
    # Enable training mode so DropBlock activates (and mc_inference=True ensures it runs at test time)
    model.model.train()

    for pass_idx in range(num_passes):
        results = model.predict(
            image_path,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=640,
            verbose=False,
        )

        frame_dets = []

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    frame_dets.append(
                        {
                            "xyxy": xyxy,
                            "class_id": class_id,
                            "class_name": get_class_name(class_id, class_names),
                            "conf": conf,
                        }
                    )

        all_passes.append(frame_dets)
        print(f"  Pass {pass_idx + 1:2d}/{num_passes}: {len(frame_dets):2d} detections")

    return all_passes


def process_mc_results(
    all_passes,
    image_path,
    class_names,
    output_dir="results/mc_inference",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Phase 3] Processing MC passes...")

    print("  [3.1] Associating detections across passes...")
    clusters = associate_detections(all_passes, iou_thresh=0.5, class_match=True)
    print(f"  ✓ Found {len(clusters)} clusters")

    print("  [3.2] Aggregating clusters...")
    merged_boxes = aggregate_clusters(clusters, all_passes, nms_thresh=0.5)
    print(f"  ✓ Aggregated to {len(merged_boxes)} boxes")

    print("  [3.3] Computing uncertainty scores...")
    for i, cluster in enumerate(clusters):
        if i >= len(merged_boxes):
            continue

        boxes_in_cluster = [all_passes[p][d]["xyxy"] for p, d in cluster]
        confs_in_cluster = [all_passes[p][d]["conf"] for p, d in cluster]

        if boxes_in_cluster:
            merged_boxes[i]["uncertainty"] = compute_calibrated_uncertainty(
                boxes_in_cluster,
                confs_in_cluster,
            )
        else:
            merged_boxes[i]["uncertainty"] = 0.0

        class_id = int(merged_boxes[i].get("class_id", 0))
        merged_boxes[i]["class_name"] = get_class_name(class_id, class_names)

    print("  ✓ Uncertainty scores computed")

    print("  [3.4] Creating visualization...")
    viz_path = draw_detections_on_images(
        image_path=str(image_path),
        detections=merged_boxes,
        output_dir=str(output_dir),
        filename_out="mc_detections_with_uncertainty.png",
    )
    print(f"  ✓ Saved visualization to {viz_path}")

    print("  [3.5] Saving detection results...")
    clean_boxes = []

    for box in merged_boxes:
        class_id = int(box.get("class_id", 0))

        clean_boxes.append(
            {
                "xyxy": [float(x) for x in box["xyxy"]],
                "class_id": class_id,
                "class_name": get_class_name(class_id, class_names),
                "conf": float(box.get("conf", 0.0)),
                "uncertainty": float(box.get("uncertainty", 0.0)),
                "num_detections": int(box.get("num_detections", 1)),
            }
        )

    detections_path = output_dir / "mc_detections.json"

    with open(detections_path, "w") as f:
        json.dump([clean_boxes], f, indent=2)

    print(f"  ✓ Saved detections to {detections_path}")

    print("\n[Summary]")
    print(f"  Input: {len(all_passes)} passes")
    print(f"  Detections per pass: {[len(p) for p in all_passes]}")
    print(f"  Merged boxes: {len(merged_boxes)}")

    if merged_boxes:
        unc_values = [float(b.get("uncertainty", 0.0)) for b in merged_boxes]
        print(f"  Uncertainty range: [{min(unc_values):.6f}, {max(unc_values):.6f}]")
    else:
        print("  Uncertainty range: N/A")

    return {
        "num_passes": len(all_passes),
        "detections_per_pass": [len(p) for p in all_passes],
        "merged_boxes": len(merged_boxes),
        "merged_detections": clean_boxes,
        "output_dir": str(output_dir),
    }


def resolve_image_from_sample(sample_token: str, camera_channel: str = "CAM_FRONT") -> str:
    """Look up the image path for a nuScenes sample via devkit."""
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-mini", dataroot="data/v1.0-mini", verbose=False)
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"][camera_channel]
    return nusc.get_sample_data_path(cam_token)


def main():
    parser = argparse.ArgumentParser(description="MC-DropBlock inference")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--image", default=None, help="Direct path to an image file")
    parser.add_argument("--sample-token", default=None, help="nuScenes sample token (uses CAM_FRONT)")
    parser.add_argument("--camera-channel", default="CAM_FRONT")
    parser.add_argument("--num-passes", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--drop-prob", type=float, default=0.03)
    parser.add_argument("--conf-thresh", type=float, default=0.2)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results/mc_inference")
    args = parser.parse_args()

    if args.sample_token:
        image_path = resolve_image_from_sample(args.sample_token, args.camera_channel)
        print(f"Sample {args.sample_token[:8]}... → {image_path}")
    elif args.image:
        image_path = args.image
    else:
        dataset_images = list(Path("data/v1.0-mini/samples").glob("*/*.jpg"))
        if not dataset_images:
            print("ERROR: No images found in data/v1.0-mini/samples/")
            sys.exit(1)
        image_path = str(dataset_images[0])
        print(f"Auto-selected image: {image_path}")

    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    print("=" * 80)
    print("MC-DROPBLOCK UNCERTAINTY ESTIMATION")
    print("=" * 80)

    print(f"\n[Step 1] Loading model: {args.model}")
    model = YOLO(args.model)
    class_names = model.names
    print("Model classes:", class_names)
    print("✓ Model loaded")

    model = inject_dropblock_into_model(
        model,
        block_size=args.block_size,
        drop_prob=args.drop_prob,
    )

    enable_mc_inference(model, enable=True)

    all_passes = run_mc_passes(
        model,
        image_path,
        class_names=class_names,
        num_passes=args.num_passes,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
    )

    results = process_mc_results(
        all_passes,
        image_path,
        class_names=class_names,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 80)
    print("✅ MC INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  • Detections: {args.output_dir}/mc_detections.json")
    print(f"  • Visualization: {args.output_dir}/mc_detections_with_uncertainty.png")

    return results


if __name__ == "__main__":
    main()