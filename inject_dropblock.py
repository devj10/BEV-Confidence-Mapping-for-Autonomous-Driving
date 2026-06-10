"""
inject_dropblock.py

Injects DropBlock2D into a YOLOv8 backbone via forward hooks without modifying
checkpoint keys, so pretrained weights load unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from dropblock import DropBlock2D

DEFAULT_BACKBONE_INDICES: tuple[int, ...] = (2, 4, 6, 8)


@dataclass
class DropBlockInjection:
    """Registered hooks + modules; call ``remove()`` to detach."""

    dropblocks: list[DropBlock2D]
    hooks: list[nn.modules.module.RemovableHandle]

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.dropblocks.clear()


def _yolo_layer_stack(yolo_model: nn.Module) -> nn.Sequential:
    """Return the indexed layer list inside an Ultralytics DetectionModel."""
    if hasattr(yolo_model, "model") and isinstance(yolo_model.model, nn.Sequential):
        return yolo_model.model
    raise TypeError(
        "Expected an Ultralytics DetectionModel with attribute ``model`` "
        f"(nn.Sequential); got {type(yolo_model).__name__}"
    )


def inject_dropblock(
    yolo_model: nn.Module,
    *,
    layer_indices: Sequence[int] | None = None,
    block_size: int = 7,
    drop_prob: float = 0.1,
    backbone_only: bool = True,
) -> DropBlockInjection:
    """Attach DropBlock2D after selected YOLO layers using forward hooks."""
    stack = _yolo_layer_stack(yolo_model)
    n_layers = len(stack)

    if layer_indices is None:
        if backbone_only:
            indices = DEFAULT_BACKBONE_INDICES
        else:
            indices = tuple(range(n_layers))
    else:
        indices = tuple(layer_indices)

    dropblocks: list[DropBlock2D] = []
    hooks: list[nn.modules.module.RemovableHandle] = []

    for idx in indices:
        if idx < 0 or idx >= n_layers:
            raise IndexError(f"Layer index {idx} out of range for model with {n_layers} layers")

        layer = stack[idx]
        db = DropBlock2D(block_size=block_size, drop_prob=drop_prob)
        dropblocks.append(db)

        def _hook(module: nn.Module, inputs: tuple, output, dropblock: DropBlock2D = db):
            if isinstance(output, torch.Tensor):
                return dropblock(output)
            if isinstance(output, (list, tuple)):
                return type(output)(
                    dropblock(t) if isinstance(t, torch.Tensor) else t for t in output
                )
            return output

        hooks.append(layer.register_forward_hook(_hook))

    yolo_model._dropblock_injection = DropBlockInjection(dropblocks=dropblocks, hooks=hooks)  # type: ignore[attr-defined]
    return yolo_model._dropblock_injection  # type: ignore[attr-defined]


def remove_dropblock(yolo_model: nn.Module) -> None:
    """Remove DropBlock hooks if present."""
    injection = getattr(yolo_model, "_dropblock_injection", None)
    if injection is not None:
        injection.remove()
        del yolo_model._dropblock_injection


def get_dropblock_injection(yolo_model: nn.Module) -> DropBlockInjection | None:
    """Return the active DropBlockInjection on a model, or None if not injected."""
    return getattr(yolo_model, "_dropblock_injection", None)
