"""Small one-dimensional convolutional baseline."""

from __future__ import annotations

import torch
from torch import nn

from ecgaf.models.pooling import masked_global_average, resize_mask
from ecgaf.models.registry import register


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int) -> None:
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.block(values)
        return output, resize_mask(mask, output.shape[-1])


@register("cnn1d")
def build_cnn1d(model_cfg: dict) -> nn.Module:
    channels = [int(channel) for channel in model_cfg.get("channels", [16, 32, 64])]
    dropout = float(model_cfg.get("dropout", 0.2))
    return SmallCNN(channels=channels, dropout=dropout)


class SmallCNN(nn.Module):
    def __init__(self, channels: list[int], dropout: float) -> None:
        super().__init__()
        blocks: list[_ConvBlock] = []
        in_channels = 1
        kernels = [7, 5, 5, 3]
        for index, out_channels in enumerate(channels):
            kernel = kernels[min(index, len(kernels) - 1)]
            blocks.append(_ConvBlock(in_channels, out_channels, kernel=kernel, stride=2))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels[-1], 4)

    def forward(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = signal.unsqueeze(1)
        for block in self.blocks:
            values, mask = block(values, mask)
        pooled = masked_global_average(values, mask)
        return self.classifier(self.dropout(pooled))
