#!/usr/bin/env python3
"""
Compute mAP for MC-DropBlock detections by fusing T passes per frame, then matching to GT.

MC inference (mc_yolo.py) writes raw per-pass boxes. This script aggregates them into one
prediction set per image and reports precision / recall / mAP@50 / mAP@50-95.

Usage:
    python eval/mc_detection_metrics.py \\
        --mc-json results/mc_raw_detections.json

    # Compare fused MC vs. using only pass 0 of each frame:
    python eval/mc_detection_metrics.py \\
        --mc-json results/mc_raw_detections.json \\
        --fusion pass0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from ultralytics.utils.metrics import DetMetrics, box_iou

IOUV = torch.linspace(0.5, 0.95, 10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mAP for fused MC-DropBlock detections")
    p.add_argument("--mc-json", required=True, help="Output JSON from mc_yolo.py")
    p.add_argument(
        "--fusion",
        choices=("union_nms", "pass0"),
        default="union_nms",
        help="union_nms: merge all T passes then NMS; pass0: first pass only",
    )
    p.add_argument("--conf", type=float, default=0.001, help="Min confidence after fusion")
    p.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU within fused boxes")
    p.add_argument("--max-images", type=int, default=None)
    return p.parse_args()


def image_to_label_path(image_path: Path) -> Path:
    parts = image_path.parts
    if "images" in parts:
        idx = parts.index("images")
        label_parts = list(parts[:idx]) + ["labels"] + list(parts[idx + 1 :])
        return Path(*label_parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO txt (cls cx cy w h norm) -> xyxy arrays."""
    if not label_path.exists():
        return np.zeros((0, 4)), np.zeros((0,))
    cls_list, boxes = [], []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        c, cx, cy, w, h = map(float, line.split())
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        cls_list.append(int(c))
        boxes.append([x1, y1, x2, y2])
    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,))
    return np.array(boxes, dtype=np.float32), np.array(cls_list, dtype=np.int64)


def fuse_union_nms(
    passes: list[dict],
    *,
    conf: float,
    nms_iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyxy, confs, clss = [], [], []
    for p in passes:
        xyxy.extend(p["xyxy"])
        confs.extend(p["conf"])
        clss.extend(p["cls"])

    if not xyxy:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    boxes = torch.tensor(xyxy, dtype=torch.float32)
    scores = torch.tensor(confs, dtype=torch.float32)
    classes = torch.tensor(clss, dtype=torch.int64)

    keep_boxes, keep_scores, keep_cls = [], [], []
    for c in classes.unique():
        mask = classes == c
        b, s = boxes[mask], scores[mask]
        if s.numel() == 0:
            continue
        idx = nms(b, s, nms_iou)
        keep_boxes.append(b[idx])
        keep_scores.append(s[idx])
        keep_cls.append(torch.full((len(idx),), c, dtype=torch.int64))

    boxes = torch.cat(keep_boxes)
    scores = torch.cat(keep_scores)
    classes = torch.cat(keep_cls)

    m = scores >= conf
    return boxes[m].numpy(), scores[m].numpy(), classes[m].numpy()


def fuse_pass0(passes: list[dict], *, conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = passes[0]
    xyxy = np.array(p["xyxy"], dtype=np.float32)
    confs = np.array(p["conf"], dtype=np.float32)
    clss = np.array(p["cls"], dtype=np.int64)
    if confs.size == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
    m = confs >= conf
    return xyxy[m], confs[m], clss[m]


def match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
) -> np.ndarray:
    """Same logic as Ultralytics DetectionValidator (10 IoU thresholds)."""
    correct = np.zeros((pred_classes.shape[0], len(IOUV)), dtype=bool)
    if pred_classes.shape[0] == 0:
        return correct
    correct_class = true_classes[:, None] == pred_classes
    iou = (iou * correct_class).cpu().numpy()
    for i, threshold in enumerate(IOUV.tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def print_metrics(metrics: DetMetrics, fusion: str) -> None:
    box = metrics.box
    names = metrics.names
    print(f"\n{'=' * 62}")
    print(f"  MC-DropBlock mAP  (fusion: {fusion})")
    print(f"  {'Class':<20s}  {'P':>6}  {'R':>6}  {'mAP@50':>8}  {'mAP@50-95':>10}")
    print("-" * 62)
    for i, cls_idx in enumerate(box.ap_class_index):
        name = names.get(cls_idx, str(cls_idx))
        p, r, ap50, ap = metrics.class_result(i)
        print(f"  {name:<20s}  {p:>6.3f}  {r:>6.3f}  {ap50:>8.3f}  {ap:>10.3f}")
    p, r, ap50, ap = metrics.mean_results()
    print("-" * 62)
    print(f"  {'all':<20s}  {p:>6.3f}  {r:>6.3f}  {ap50:>8.3f}  {ap:>10.3f}")
    print("=" * 62 + "\n")


def main() -> None:
    args = parse_args()
    mc_path = Path(args.mc_json)
    if not mc_path.exists():
        print(f"ERROR: not found: {mc_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(mc_path.read_text())
    frames = data["frames"]
    if args.max_images:
        frames = frames[: args.max_images]

    names = {int(k): v for k, v in data.get("class_names", {}).items()}
    metrics = DetMetrics(names)

    for im_idx, frame in enumerate(frames):
        img_path = Path(frame["image"])
        if not img_path.is_absolute():
            img_path = Path.cwd() / img_path
        if not img_path.exists():
            print(f"WARNING: skip missing image {img_path}", file=sys.stderr)
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        gt_xyxy, gt_cls = load_gt_boxes(image_to_label_path(img_path), w, h)
        if args.fusion == "pass0":
            pred_xyxy, pred_conf, pred_cls = fuse_pass0(frame["passes"], conf=args.conf)
        else:
            pred_xyxy, pred_conf, pred_cls = fuse_union_nms(
                frame["passes"], conf=args.conf, nms_iou=args.nms_iou
            )

        gt_t = torch.tensor(gt_xyxy, dtype=torch.float32)
        gt_c = torch.tensor(gt_cls, dtype=torch.int64)
        pred_t = torch.tensor(pred_xyxy, dtype=torch.float32)
        pred_c = torch.tensor(pred_cls, dtype=torch.int64)
        pred_conf_t = torch.tensor(pred_conf, dtype=torch.float32)

        if pred_t.shape[0] == 0:
            tp = np.zeros((0, len(IOUV)), dtype=bool)
        elif gt_t.shape[0] == 0:
            tp = np.zeros((pred_t.shape[0], len(IOUV)), dtype=bool)
        else:
            iou = box_iou(gt_t, pred_t)
            tp = match_predictions(pred_c, gt_c, iou)

        target_cls = gt_c.numpy() if gt_c.numel() else np.zeros((0,))
        metrics.update_stats(
            {
                "tp": tp,
                "conf": pred_conf_t.numpy(),
                "pred_cls": pred_c.numpy(),
                "target_cls": target_cls,
                "target_img": np.unique(gt_c.numpy()) if gt_c.numel() else np.zeros((0,)),
                "im_name": str(img_path),
            }
        )

    metrics.process(plot=False)
    print_metrics(metrics, args.fusion)
    print(
        "Compare to baseline (no MC, standard val): ~0.337 mAP@50, ~0.175 mAP@50-95\n"
        "  python eval/detection_metrics.py --weights <best.pt> --data data/yolo_out/dataset.yaml"
    )


if __name__ == "__main__":
    main()
