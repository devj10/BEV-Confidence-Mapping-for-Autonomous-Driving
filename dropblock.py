"""
DropBlock2D for MC-DropBlock uncertainty estimation.

During standard eval, activations pass through unchanged. With ``mc_inference=True``
(or ``model.train()``), spatial blocks are dropped stochastically on each forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2D(nn.Module):
    """Drop contiguous spatial blocks (better than dropout on conv feature maps)."""

    def __init__(self, block_size: int = 7, drop_prob: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
        self.mc_inference = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and not self.mc_inference:
            return x

        B, C, H, W = x.shape
        if H < self.block_size or W < self.block_size:
            return x

        mask_h = H - self.block_size + 1
        mask_w = W - self.block_size + 1

        p = self.drop_prob / (self.block_size**2)
        mask = torch.bernoulli(torch.full((B, C, mask_h, mask_w), p, device=x.device, dtype=x.dtype))

        # Expand seed mask to block mask; pad/crop so spatial dims match x exactly.
        pad = self.block_size // 2
        mask = F.pad(mask, (pad, pad, pad, pad), value=0)
        mask = F.max_pool2d(
            mask,
            kernel_size=(self.block_size, self.block_size),
            stride=1,
            padding=0,
        )
        if mask.shape[-2] < H or mask.shape[-1] < W:
            mask = F.pad(mask, (0, W - mask.shape[-1], 0, H - mask.shape[-2]))
        mask = mask[:, :, :H, :W]

        mask = 1 - mask
        mask = mask * (mask.numel() / mask.sum().clamp(min=1))
        return x * mask


def iter_dropblocks(model: nn.Module):
    """Yield every DropBlock2D submodule."""
    for module in model.modules():
        if isinstance(module, DropBlock2D):
            yield module


def set_mc_inference(model: nn.Module, enabled: bool) -> None:
    """Toggle stochastic DropBlock for MC inference while keeping BN in eval mode."""
    for db in iter_dropblocks(model):
        db.mc_inference = enabled


def enable_mc_inference(model: nn.Module) -> None:
    set_mc_inference(model, True)


def disable_mc_inference(model: nn.Module) -> None:
    set_mc_inference(model, False)
