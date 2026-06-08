#!/usr/bin/env python3
"""
run_mc_dropblock_inference.py

MC-DropBlock uncertainty inference for a fine-tuned YOLOv8 model.

Supports:
  1. Single image:
      python scripts/run_mc_dropblock_inference.py \
        --model model_final.pt \
        --image path/to/image.jpg \
        --num-passes 10 \
        --output-dir results/mc_single

  2. Single nuScenes sample:
      python scripts/run_mc_dropblock_inference.py \
        --model model_final.pt \
        --sample-token SAMPLE_TOKEN \
        --version v1.0-mini \
        --dataroot data/v1.0-mini \
        --camera-channel CAM_FRONT \
        --num-passes 10 \
        --output-dir results/mc_sample

  3. Multiple nuScenes scenes:
      python scripts/run_mc_dropblock_inference.py \
        --model model_final.pt \
        --scene-names scene-0061 scene-0062 \
        --version v1.0-mini \
        --dataroot data/v1.0-mini \
        --camera-channel CAM_FRONT \
        --num-passes 10 \
        --max-frames 5 \
        --output-dir results/mc_scenes
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

        # Preserve Ultralytics routing metadata.
        self.f = layer.f
        self.i = layer.i
        self.type = layer.type
        self.np = getattr(layer, "np", 0)

    def forward(self, x):
        return self.db(self.layer(x))


def parse_args():
    parser = argparse.ArgumentParser(description="MC-DropBlock inference")

    parser.add_argument("--model", default="yolov8n.pt")

    # Input modes
    parser.add_argument("--image", default=None, help="Direct path to one image file")
    parser.add_argument("--sample-token", default=None, help="nuScenes sample token")
    parser.add_argument(
        "--scene-names",
        nargs="+",
        default=None,
        help="One or more nuScenes scene names, e.g. scene-0061 scene-0062",
    )

    # nuScenes
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--dataroot", default="data/v1.0-mini")
    parser.add_argument("--camera-channel", default="CAM_FRONT")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional max frames per scene",
    )

    # MC inference
    parser.add_argument("--num-passes", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--drop-prob", type=float, default=0.03)
    parser.add_argument("--conf-thresh", type=float, default=0.2)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="'0', 'cpu', etc.")

    # Output
    parser.add_argument("--output-dir", default="results/mc_inference")
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization image output",
    )

    return parser.parse_args()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    return obj


def get_class_name(class_id, class_names):
    class_id = int(class_id)

    if isinstance(class_names, dict):
        return str(class_names.get(class_id, f"class_{class_id}"))

    if isinstance(class_names, list):
        if 0 <= class_id < len(class_names):
            return str(class_names[class_id])

    return f"class_{class_id}"


def inject_dropblock_into_model(model, block_size=3, drop_prob=0.03):
    print(
        f"\n[MC-DropBlock] Injecting DropBlock "
        f"(block_size={block_size}, drop_prob={drop_prob})"
    )

    model_net = model.model
    injected_count = 0

    # Safer later YOLO blocks; avoid early backbone convs.
    safe_layer_indices = [12, 15, 18]

    for idx in safe_layer_indices:
        if idx >= len(model_net.model):
            continue

        old_layer = model_net.model[idx]

        # Avoid double-wrapping if script/function is reused.
        if isinstance(old_layer, YOLOLayerWithDropBlock):
            continue

        db = DropBlock2D(block_size=block_size, drop_prob=drop_prob)
        model_net.model[idx] = YOLOLayerWithDropBlock(old_layer, db)
        injected_count += 1

    print(f"✓ Injected DropBlock into {injected_count} YOLO blocks")
    return model


def enable_mc_inference(model, enable=True):
    """
    Enable stochastic DropBlock during inference while keeping BatchNorm frozen.

    Do NOT call model.model.train() globally, because that also puts BN layers
    in train mode. Instead:
      - whole model eval
      - DropBlock train / mc_inference enabled
      - BatchNorm eval
    """
    model.model.eval()

    for module in model.model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

        if isinstance(module, DropBlock2D):
            module.mc_inference = enable
            if enable:
                module.train()
            else:
                module.eval()

    if enable:
        print("[MC-DropBlock] MC inference mode ENABLED")
    else:
        print("[MC-DropBlock] MC inference mode DISABLED")


def load_model(args):
    print("=" * 80)
    print("MC-DROPBLOCK UNCERTAINTY ESTIMATION")
    print("=" * 80)

    print(f"\n[Step 1] Loading model: {args.model}")
    model = YOLO(args.model)

    if args.device is not None:
        model.to(args.device)

    class_names = model.names
    print("Model classes:", class_names)
    print("✓ Model loaded")

    model = inject_dropblock_into_model(
        model,
        block_size=args.block_size,
        drop_prob=args.drop_prob,
    )

    enable_mc_inference(model, enable=True)

    return model, class_names


def extract_detections_from_yolo_result(result, class_names):
    frame_dets = []

    if result.boxes is None:
        return frame_dets

    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
        conf = float(box.conf[0].detach().cpu().item())
        class_id = int(box.cls[0].detach().cpu().item())

        frame_dets.append(
            {
                "xyxy": [float(x) for x in xyxy],
                "class_id": class_id,
                "class_name": get_class_name(class_id, class_names),
                "conf": conf,
            }
        )

    return frame_dets


def run_mc_passes(
    model,
    image_path,
    class_names,
    num_passes=20,
    conf_thresh=0.3,
    iou_thresh=0.5,
    imgsz=640,
    device=None,
):
    print(f"\n[MC Inference] Running {num_passes} stochastic forward passes...")

    all_passes = []

    with torch.no_grad():
        for pass_idx in range(num_passes):
            # Ultralytics may internally touch modes, so re-enable each pass.
            enable_mc_inference(model, enable=True)

            results = model.predict(
                image_path,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            frame_dets = []

            if results and len(results) > 0:
                frame_dets = extract_detections_from_yolo_result(
                    results[0],
                    class_names,
                )

            all_passes.append(frame_dets)
            print(
                f"  Pass {pass_idx + 1:2d}/{num_passes}: "
                f"{len(frame_dets):2d} detections"
            )

    return all_passes


def process_mc_results(
    all_passes,
    image_path,
    class_names,
    output_dir="results/mc_inference",
    nms_thresh=0.5,
    cluster_iou_thresh=0.5,
    make_viz=True,
    filename_prefix="mc",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Phase 3] Processing MC passes...")

    print("  [3.1] Associating detections across passes...")
    clusters = associate_detections(
        all_passes,
        iou_thresh=cluster_iou_thresh,
        class_match=True,
    )
    print(f"  ✓ Found {len(clusters)} clusters")

    print("  [3.2] Aggregating clusters...")
    merged_boxes = aggregate_clusters(
        clusters,
        all_passes,
        nms_thresh=nms_thresh,
    )
    print(f"  ✓ Aggregated to {len(merged_boxes)} boxes")

    print("  [3.3] Computing uncertainty scores...")
    for i, cluster in enumerate(clusters):
        if i >= len(merged_boxes):
            continue

        boxes_in_cluster = []
        confs_in_cluster = []

        for p, d in cluster:
            if p < len(all_passes) and d < len(all_passes[p]):
                boxes_in_cluster.append(all_passes[p][d]["xyxy"])
                confs_in_cluster.append(all_passes[p][d]["conf"])

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

    if make_viz:
        print("  [3.4] Creating visualization...")
        viz_name = f"{filename_prefix}_detections_with_uncertainty.png"
        viz_path = draw_detections_on_images(
            image_path=str(image_path),
            detections=clean_boxes,
            output_dir=str(output_dir),
            filename_out=viz_name,
        )
        print(f"  ✓ Saved visualization to {viz_path}")

    print("  [3.5] Saving detection results...")
    detections_path = output_dir / f"{filename_prefix}_detections.json"

    with open(detections_path, "w") as f:
        json.dump(json_safe(clean_boxes), f, indent=2)

    print(f"  ✓ Saved detections to {detections_path}")

    summary = {
        "image_path": str(image_path),
        "num_passes": len(all_passes),
        "detections_per_pass": [len(p) for p in all_passes],
        "num_clusters": len(clusters),
        "merged_boxes": len(clean_boxes),
        "merged_detections": clean_boxes,
        "output_dir": str(output_dir),
    }

    if clean_boxes:
        unc_values = [float(b.get("uncertainty", 0.0)) for b in clean_boxes]
        summary["uncertainty_min"] = min(unc_values)
        summary["uncertainty_max"] = max(unc_values)
        summary["uncertainty_mean"] = sum(unc_values) / len(unc_values)
    else:
        summary["uncertainty_min"] = None
        summary["uncertainty_max"] = None
        summary["uncertainty_mean"] = None

    summary_path = output_dir / f"{filename_prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("\n[Summary]")
    print(f"  Input: {len(all_passes)} passes")
    print(f"  Detections per pass: {[len(p) for p in all_passes]}")
    print(f"  Merged boxes: {len(clean_boxes)}")

    if clean_boxes:
        print(
            f"  Uncertainty range: "
            f"[{summary['uncertainty_min']:.6f}, {summary['uncertainty_max']:.6f}]"
        )
    else:
        print("  Uncertainty range: N/A")

    return summary


def make_nuscenes(version: str, dataroot: str):
    from nuscenes.nuscenes import NuScenes

    return NuScenes(version=version, dataroot=dataroot, verbose=False)


def resolve_image_from_sample(
    nusc,
    sample_token: str,
    camera_channel: str = "CAM_FRONT",
) -> str:
    sample = nusc.get("sample", sample_token)

    if camera_channel not in sample["data"]:
        raise KeyError(
            f"Camera channel {camera_channel} not found in sample data. "
            f"Available: {list(sample['data'].keys())}"
        )

    cam_token = sample["data"][camera_channel]
    return nusc.get_sample_data_path(cam_token)


def get_scene_sample_tokens(nusc, scene_name: str):
    scene = next((s for s in nusc.scene if s["name"] == scene_name), None)

    if scene is None:
        available = [s["name"] for s in nusc.scene[:20]]
        raise ValueError(
            f"Scene not found: {scene_name}. "
            f"First available scenes: {available}"
        )

    tokens = []
    sample_token = scene["first_sample_token"]

    while sample_token:
        sample = nusc.get("sample", sample_token)
        tokens.append(sample_token)
        sample_token = sample["next"]

    return tokens


def process_single_image(
    model,
    image_path,
    class_names,
    args,
    output_dir,
    filename_prefix="mc",
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    all_passes = run_mc_passes(
        model,
        str(image_path),
        class_names=class_names,
        num_passes=args.num_passes,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        imgsz=args.imgsz,
        device=args.device,
    )

    return process_mc_results(
        all_passes,
        str(image_path),
        class_names=class_names,
        output_dir=output_dir,
        nms_thresh=args.nms_thresh,
        cluster_iou_thresh=args.iou_thresh,
        make_viz=not args.no_viz,
        filename_prefix=filename_prefix,
    )


def process_scene(
    model,
    nusc,
    scene_name,
    class_names,
    args,
):
    scene_output_dir = Path(args.output_dir) / scene_name
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    sample_tokens = get_scene_sample_tokens(nusc, scene_name)

    if args.max_frames is not None:
        sample_tokens = sample_tokens[: args.max_frames]

    print(f"\n[Scene] {scene_name}: {len(sample_tokens)} frames")

    scene_results = []

    for frame_idx, sample_token in enumerate(sample_tokens):
        image_path = resolve_image_from_sample(
            nusc,
            sample_token,
            camera_channel=args.camera_channel,
        )

        if not Path(image_path).exists():
            print(f"  Skipping missing image: {image_path}")
            continue

        frame_prefix = f"{frame_idx:04d}_{sample_token[:8]}"
        frame_output_dir = scene_output_dir / frame_prefix

        print(
            f"\n[{scene_name}] Frame {frame_idx + 1}/{len(sample_tokens)} "
            f"{sample_token[:8]} → {image_path}"
        )

        summary = process_single_image(
            model=model,
            image_path=image_path,
            class_names=class_names,
            args=args,
            output_dir=frame_output_dir,
            filename_prefix=frame_prefix,
        )

        summary["scene_name"] = scene_name
        summary["sample_token"] = sample_token
        summary["camera_channel"] = args.camera_channel
        summary["frame_idx"] = frame_idx

        scene_results.append(summary)

    scene_summary_path = scene_output_dir / "scene_mc_summary.json"

    with open(scene_summary_path, "w") as f:
        json.dump(json_safe(scene_results), f, indent=2)

    print(f"\nSaved scene summary to {scene_summary_path}")

    return scene_results


def choose_default_image():
    dataset_images = list(Path("data/v1.0-mini/samples").glob("*/*.jpg"))

    if not dataset_images:
        raise FileNotFoundError("No images found in data/v1.0-mini/samples/")

    image_path = str(dataset_images[0])
    print(f"Auto-selected image: {image_path}")
    return image_path


def main():
    args = parse_args()

    if args.num_passes <= 0:
        raise ValueError("--num-passes must be positive")

    model, class_names = load_model(args)

    # Multi-scene mode
    if args.scene_names:
        nusc = make_nuscenes(args.version, args.dataroot)

        all_scene_results = {}

        for scene_name in args.scene_names:
            all_scene_results[scene_name] = process_scene(
                model=model,
                nusc=nusc,
                scene_name=scene_name,
                class_names=class_names,
                args=args,
            )

        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        all_summary_path = output_root / "all_scenes_mc_summary.json"

        with open(all_summary_path, "w") as f:
            json.dump(json_safe(all_scene_results), f, indent=2)

        print("\n" + "=" * 80)
        print("✅ MC SCENE INFERENCE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Saved all-scene summary to: {all_summary_path}")

        return all_scene_results

    # Single sample-token mode
    if args.sample_token:
        nusc = make_nuscenes(args.version, args.dataroot)
        image_path = resolve_image_from_sample(
            nusc,
            args.sample_token,
            args.camera_channel,
        )
        print(f"Sample {args.sample_token[:8]}... → {image_path}")

    # Single image mode
    elif args.image:
        image_path = args.image

    # Default mini image
    else:
        image_path = choose_default_image()

    summary = process_single_image(
        model=model,
        image_path=image_path,
        class_names=class_names,
        args=args,
        output_dir=args.output_dir,
        filename_prefix="mc",
    )

    print("\n" + "=" * 80)
    print("✅ MC INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  • Detections: {args.output_dir}/mc_detections.json")
    print(f"  • Summary:    {args.output_dir}/mc_summary.json")
    if not args.no_viz:
        print(f"  • Visualization: {args.output_dir}/mc_detections_with_uncertainty.png")

    return summary


if __name__ == "__main__":
    main()