"""Heads that sit on the small dilated residual encoder."""

from __future__ import annotations

import math

import torch
from torch import nn

from ecgaf.models.dilated_resnet import DilatedResNet
from ecgaf.models.pooling import masked_global_average, resize_mask
from ecgaf.models.registry import register


def _encoder(model_cfg: dict) -> DilatedResNet:
    channels = [int(channel) for channel in model_cfg.get("channels", [16, 32, 64, 128])]
    dilations = [int(dilation) for dilation in model_cfg.get("dilations", [1, 2, 4, 8])]
    if len(dilations) != len(channels):
        raise ValueError("dilations and channels must have the same length")
    return DilatedResNet(channels, dilations, dropout=0.0, with_head=False)


class _PositionalEncoding(nn.Module):
    def __init__(self, channels: int, max_length: int = 4000) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, channels, 2, dtype=torch.float32) * (-math.log(10000.0) / channels))
        encoding = torch.zeros(max_length, channels)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence + self.encoding[:, : sequence.shape[1]]


def _valid_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
    resized = resize_mask(mask, length)
    return resized.squeeze(1) >= 0.5


@register("dilated_resnet_rr")
def build_rr(model_cfg: dict) -> nn.Module:
    return DilatedResNetRR(model_cfg)


@register("dilated_resnet_gru")
def build_gru(model_cfg: dict) -> nn.Module:
    return DilatedResNetGRU(model_cfg)


@register("dilated_resnet_transformer")
def build_transformer(model_cfg: dict) -> nn.Module:
    return DilatedResNetTransformer(model_cfg)


class DilatedResNetRR(nn.Module):
    """Waveform embedding plus a short RR-interval vector."""

    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        self.encoder = _encoder(model_cfg)
        dropout = float(model_cfg.get("dropout", 0.2))
        rr_dim = int(model_cfg.get("rr_dim", 16))
        hidden = int(model_cfg.get("rr_hidden", 32))
        channels = [int(channel) for channel in model_cfg.get("channels", [16, 32, 64, 128])]
        wave_dim = channels[-1]
        self.rr_net = nn.Sequential(
            nn.Linear(rr_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(wave_dim + hidden, 4)

    def forward(self, signal: torch.Tensor, mask: torch.Tensor, extras: torch.Tensor | None = None) -> torch.Tensor:
        if extras is None:
            raise ValueError("The RR fusion model needs the RR feature vector")
        values, encoded_mask = self.encoder.encode(signal, mask)
        waveform = masked_global_average(values, encoded_mask)
        rhythm = self.rr_net(extras)
        return self.classifier(self.dropout(torch.cat([waveform, rhythm], dim=-1)))


class DilatedResNetGRU(nn.Module):
    """A small GRU reads the encoder sequence so beat order is explicit."""

    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        self.encoder = _encoder(model_cfg)
        dropout = float(model_cfg.get("dropout", 0.2))
        channels = [int(channel) for channel in model_cfg.get("channels", [16, 32, 64, 128])]
        hidden = int(model_cfg.get("gru_hidden", 64))
        self.gru = nn.GRU(
            input_size=channels[-1],
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, 4)

    def forward(self, signal: torch.Tensor, mask: torch.Tensor, extras: torch.Tensor | None = None) -> torch.Tensor:
        del extras
        values, encoded_mask = self.encoder.encode(signal, mask)
        sequence = values.transpose(1, 2)
        outputs, _hidden = self.gru(sequence)
        valid = _valid_mask(encoded_mask, outputs.shape[1]).unsqueeze(-1).to(outputs.dtype)
        pooled = (outputs * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))


class DilatedResNetTransformer(nn.Module):
    """A small transformer encoder on the already-short convolutional sequence."""

    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        self.encoder = _encoder(model_cfg)
        dropout = float(model_cfg.get("dropout", 0.2))
        channels = [int(channel) for channel in model_cfg.get("channels", [16, 32, 64, 128])]
        width = channels[-1]
        heads = int(model_cfg.get("transformer_heads", 4))
        layers = int(model_cfg.get("transformer_layers", 2))
        feedforward = int(model_cfg.get("transformer_ff", 256))
        self.position = _PositionalEncoding(width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, 4)

    def forward(self, signal: torch.Tensor, mask: torch.Tensor, extras: torch.Tensor | None = None) -> torch.Tensor:
        del extras
        values, encoded_mask = self.encoder.encode(signal, mask)
        sequence = self.position(values.transpose(1, 2))
        valid = _valid_mask(encoded_mask, sequence.shape[1])
        padding = ~valid
        if padding.all(dim=1).any():
            padding = padding.clone()
            padding[:, 0] = False
        outputs = self.transformer(sequence, src_key_padding_mask=padding)
        weights = valid.unsqueeze(-1).to(outputs.dtype)
        pooled = (outputs * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))
