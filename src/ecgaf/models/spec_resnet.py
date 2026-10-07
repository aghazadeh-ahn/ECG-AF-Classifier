"""Dilated residual network on a log spectrogram. Time is dilated; frequency is local."""

from __future__ import annotations

import torch
from torch import nn

from ecgaf.models.pooling import resize_mask
from ecgaf.models.registry import register


class _ResidualStage2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dilation: int) -> None:
        super().__init__()
        padding = (1, time_dilation)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=padding,
            dilation=(1, time_dilation),
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=(1, time_dilation),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.skip(values)
        output = self.relu(self.bn1(self.conv1(values)))
        output = self.bn2(self.conv2(output))
        freq = min(output.shape[-2], residual.shape[-2])
        time = min(output.shape[-1], residual.shape[-1])
        output = self.relu(output[..., :freq, :time] + residual[..., :freq, :time])
        return output, resize_mask(mask, time)


@register("spec_resnet")
def build_spec_resnet(model_cfg: dict) -> nn.Module:
    channels = [int(channel) for channel in model_cfg.get("channels", [16, 32, 64, 128])]
    dilations = [int(dilation) for dilation in model_cfg.get("dilations", [1, 2, 4, 8])]
    if len(dilations) != len(channels):
        raise ValueError("dilations and channels must have the same length")
    dropout = float(model_cfg.get("dropout", 0.2))
    return SpecResNet(channels=channels, dilations=dilations, dropout=dropout)


class SpecResNet(nn.Module):
    def __init__(self, channels: list[int], dilations: list[int], dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        stages: list[_ResidualStage2d] = []
        in_channels = channels[0]
        for index, out_channels in enumerate(channels):
            stages.append(_ResidualStage2d(in_channels, out_channels, dilations[index]))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.dilations = list(dilations)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels[-1], 4)

    def forward(self, spectrum: torch.Tensor, mask: torch.Tensor, extras: torch.Tensor | None = None) -> torch.Tensor:
        del extras
        values = self.stem(spectrum.unsqueeze(1))
        mask = resize_mask(mask, values.shape[-1])
        for stage in self.stages:
            values, mask = stage(values, mask)
        pooled = _masked_average(values, mask)
        return self.classifier(self.dropout(pooled))


def _masked_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """values is (batch, channels, frequency, time)."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    mask = resize_mask(mask, values.shape[-1]).to(dtype=values.dtype)
    weighted = values * mask.unsqueeze(2)
    denominator = mask.sum(dim=-1).clamp(min=1.0) * values.shape[-2]
    return weighted.sum(dim=(-1, -2)) / denominator
