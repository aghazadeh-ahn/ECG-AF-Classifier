"""Dilated residual network. Dilation is the part that sees rhythm, not only local shape."""

from __future__ import annotations

import torch
from torch import nn

from ecgaf.models.pooling import MaskedTemporalAttention, masked_global_average, resize_mask
from ecgaf.models.registry import register


class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.skip(values)
        output = self.relu(self.bn1(self.conv1(values)))
        output = self.bn2(self.conv2(output))
        length = min(output.shape[-1], residual.shape[-1])
        output = self.relu(output[..., :length] + residual[..., :length])
        return output, resize_mask(mask, length)


@register("dilated_resnet")
def build_dilated_resnet(model_cfg: dict) -> nn.Module:
    channels = [int(channel) for channel in model_cfg.get("channels", [16, 32, 64, 128])]
    dilations = [int(dilation) for dilation in model_cfg.get("dilations", [1, 2, 4, 8])]
    if len(dilations) != len(channels):
        raise ValueError("dilations and channels must have the same length")
    dropout = float(model_cfg.get("dropout", 0.2))
    pooling = str(model_cfg.get("pooling", "mean"))
    return DilatedResNet(
        channels=channels,
        dilations=dilations,
        dropout=dropout,
        pooling=pooling,
    )


class DilatedResNet(nn.Module):
    def __init__(
        self,
        channels: list[int],
        dilations: list[int],
        dropout: float,
        pooling: str = "mean",
        with_head: bool = True,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError(f"Unknown pooling mode: {pooling}")
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        stages: list[ResidualStage] = []
        in_channels = channels[0]
        for index, out_channels in enumerate(channels):
            stages.append(
                ResidualStage(
                    in_channels,
                    out_channels,
                    stride=2,
                    dilation=dilations[index],
                )
            )
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.dilations = list(dilations)
        self.pooling_name = pooling
        self.attention = MaskedTemporalAttention(channels[-1]) if with_head and pooling == "attention" else None
        self.dropout = nn.Dropout(dropout) if with_head else None
        self.classifier = nn.Linear(channels[-1], 4) if with_head else None

    def encode(self, signal: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.stem(signal.unsqueeze(1))
        mask = resize_mask(mask, values.shape[-1])
        for stage in self.stages:
            values, mask = stage(values, mask)
        return values, mask

    def forward(self, signal: torch.Tensor, mask: torch.Tensor, extras: torch.Tensor | None = None) -> torch.Tensor:
        del extras
        values, mask = self.encode(signal, mask)
        if self.attention is None:
            pooled = masked_global_average(values, mask)
        else:
            pooled = self.attention(values, mask)
        return self.classifier(self.dropout(pooled))
