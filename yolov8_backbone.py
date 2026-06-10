"""
yolov8_backbone.py

Inserts DropBlock2D after each C2f block's cv2 conv in a YOLOv8 backbone.
"""

from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f, Bottleneck
from mc_dropblock import DropBlock2D

def inject_dropblock(model, block_size=7, drop_prob=0.1):
    """Walk the model and insert DropBlock2D after each C2f block's cv2 conv."""
    for name, module in model.named_modules():
        if isinstance(module, C2f):
            original_cv2 = module.cv2
            module.cv2 = nn.Sequential(
                original_cv2,
                DropBlock2D(block_size=block_size, drop_prob=drop_prob)
            )
    return model

yolo = YOLO("yolov8m.pt")
inject_dropblock(yolo.model, block_size=7, drop_prob=0.1)
