import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock2D(nn.Module):
    """
    Drops contiguous spatial blocks instead of random individual neurons.
    Better than Dropout for conv layers because it breaks spatial correlation.
    """
    def __init__(self, block_size=7, drop_prob=0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        # If not training AND not in MC mode, pass through unchanged
        if not self.training:
            return x

        B, C, H, W = x.shape
        # Compute mask at reduced spatial size, then expand
        mask_h = H - self.block_size + 1
        mask_w = W - self.block_size + 1

        # Probability of each seed being dropped
        p = self.drop_prob / (self.block_size ** 2)
        mask = torch.bernoulli(torch.ones(B, C, mask_h, mask_w, device=x.device) * p)

        # Expand seed mask into full blocks via max_pool
        mask = F.max_pool2d(
            mask,
            kernel_size=(self.block_size, self.block_size),
            stride=1,
            padding=self.block_size // 2
        )
        # Crop back to original spatial size if needed
        mask = mask[:, :, :H, :W]

        # Invert (1 = keep, 0 = drop) and normalize
        mask = 1 - mask
        mask = mask * (mask.numel() / mask.sum().clamp(min=1))
        return x * mask