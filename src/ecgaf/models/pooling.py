"""Pool over time while ignoring padded samples."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    if mask.shape[-1] == length:
        return mask
    return F.adaptive_max_pool1d(mask, length)


def masked_global_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """values is (batch, channels, time). Padding marked 0 in the mask is dropped."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    mask = resize_mask(mask, values.shape[-1]).to(dtype=values.dtype)
    weighted = values * mask
    denominator = mask.sum(dim=-1).clamp(min=1.0)
    return weighted.sum(dim=-1) / denominator


class MaskedTemporalAttention(torch.nn.Module):
    """One lightweight score per time step, then a masked weighted average."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = torch.nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        mask = resize_mask(mask, values.shape[-1]).to(dtype=values.dtype)
        logits = self.score(values)
        logits = logits.masked_fill(mask < 0.5, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        return (values * weights).sum(dim=-1)
