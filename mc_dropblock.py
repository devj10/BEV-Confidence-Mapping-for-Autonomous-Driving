"""Backward-compatible re-export; prefer ``dropblock``."""

from dropblock import DropBlock2D, disable_mc_inference, enable_mc_inference, set_mc_inference

__all__ = ["DropBlock2D", "disable_mc_inference", "enable_mc_inference", "set_mc_inference"]
