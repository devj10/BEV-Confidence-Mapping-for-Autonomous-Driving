#!/usr/bin/env python3
"""
mc_yolo.py

Runs T stochastic forward passes per frame using MC-DropBlock (BatchNorm in eval mode,
DropBlock active) and writes raw per-pass detections to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm
from ultralytics import YOLO

from dropblock import disable_mc_inference, enable_mc_inference, set_mc_inference
from inject_dropblock import inject_dropblock, remove_dropblock


def _boxes_from_result(result) -> dict[str, list]:
    """Extract xyxy boxes, scores, and class ids from one Ultralytics Result."""
    if result.boxes is None or len(result.boxes) == 0:
        return {"xyxy": [], "conf": [], "cls": []}

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy().tolist()
    conf = boxes.conf.cpu().numpy().tolist()
    cls = boxes.cls.cpu().numpy().astype(int).tolist()
    return {"xyxy": xyxy, "conf": conf, "cls": cls}


def run_mc_on_frame(
    model: YOLO,
    image_path: str | Path,
    *,
    T: int = 20,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str | None = None,
) -> list[dict[str, list]]:
    """Run T MC passes on a single image and return T detection dicts."""
    passes: list[dict[str, list]] = []
    predict_kwargs: dict[str, Any] = dict(
        source=str(image_path),
        stream=False,
        verbose=False,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
    )
    if device is not None:
        predict_kwargs["device"] = device

    for _ in range(T):
        results = model.predict(**predict_kwargs)
        passes.append(_boxes_from_result(results[0]))
    return passes


def collect_image_paths(source: str | Path) -> list[Path]:
    """Return a sorted list of image paths from a file or directory."""
    source = Path(source)
    if source.is_file():
        return [source]
    if source.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in exts)
        if not paths:
            raise FileNotFoundError(f"No images found under {source}")
        return paths
    raise FileNotFoundError(f"Source not found: {source}")


def run_mc_dataset(
    model: YOLO,
    image_paths: list[Path],
    *,
    T: int = 20,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Run MC inference over a list of image paths and return per-frame records."""
    records: list[dict[str, Any]] = []
    for path in tqdm(image_paths, desc="MC inference", unit="frame"):
        passes = run_mc_on_frame(
            model,
            path,
            T=T,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )
        records.append({"image": str(path), "T": T, "passes": passes})
    return records


def load_mc_yolo(
    weights: str | Path,
    *,
    block_size: int = 7,
    drop_prob: float = 0.1,
    layer_indices: tuple[int, ...] | None = None,
    mc_inference: bool = True,
) -> YOLO:
    """Load YOLO, inject DropBlock hooks, and set MC inference mode."""
    yolo = YOLO(str(weights))
    inject_dropblock(
        yolo.model,
        layer_indices=layer_indices,
        block_size=block_size,
        drop_prob=drop_prob,
    )
    yolo.model.eval()
    if mc_inference:
        enable_mc_inference(yolo.model)
    else:
        disable_mc_inference(yolo.model)
    return yolo


def set_inference_mode(model: YOLO, *, mc: bool) -> None:
    """Toggle MC-DropBlock on or off while keeping BatchNorm in eval mode."""
    model.model.eval()
    set_mc_inference(model.model, mc)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for weights, source, MC passes, and output path."""
    parser = argparse.ArgumentParser(description="MC-DropBlock YOLO inference (T passes per frame)")
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt checkpoint")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--out", required=True, help="Output JSON path for raw per-pass detections")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--drop-prob", type=float, default=0.1)
    parser.add_argument("--deterministic", action="store_true",
                        help="Single deterministic pass (DropBlock disabled)")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Load model, run MC inference over the source images, and write results to JSON."""
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        sys.exit(1)

    image_paths = collect_image_paths(args.source)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    T = 1 if args.deterministic else args.T
    yolo = load_mc_yolo(
        weights,
        block_size=args.block_size,
        drop_prob=args.drop_prob,
        mc_inference=not args.deterministic,
    )

    try:
        records = run_mc_dataset(
            yolo,
            image_paths,
            T=T,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
        )
    finally:
        remove_dropblock(yolo.model)

    payload = {
        "weights":     str(weights.resolve()),
        "source":      str(Path(args.source).resolve()),
        "T":           T,
        "mc_inference": not args.deterministic,
        "conf":        args.conf,
        "iou":         args.iou,
        "imgsz":       args.imgsz,
        "block_size":  args.block_size,
        "drop_prob":   args.drop_prob,
        "class_names": {int(k): v for k, v in yolo.names.items()},
        "frames":      records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(records)} frames × {T} passes → {out_path}")


if __name__ == "__main__":
    main()
