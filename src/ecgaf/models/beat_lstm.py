"""LSTM over heartbeats. Mean pooling keeps the rhythm; max pooling keeps one abnormal beat."""

from __future__ import annotations

import torch
from torch import nn

from ecgaf.models.registry import register


@register("beat_lstm")
def build_beat_lstm(model_cfg: dict) -> nn.Module:
    return BeatLSTM(
        beat_length=int(model_cfg.get("beat_length", 180)),
        hidden=int(model_cfg.get("hidden", 128)),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )


class BeatLSTM(nn.Module):
    def __init__(self, beat_length: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.beat_length = beat_length
        self.wave = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lstm = nn.LSTM(input_size=34, hidden_size=hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, 4)

    def forward(self, beats: torch.Tensor, mask: torch.Tensor, extras: torch.Tensor | None = None) -> torch.Tensor:
        del extras
        batch, length, _features = beats.shape
        waveform = beats[..., : self.beat_length].reshape(batch * length, 1, self.beat_length)
        rhythm = beats[..., self.beat_length :]
        embedding = self.wave(waveform).reshape(batch, length, 32)
        outputs, _state = self.lstm(torch.cat([embedding, rhythm], dim=-1))
        valid = mask >= 0.5
        if (~valid).all(dim=1).any():
            valid = valid.clone()
            valid[:, 0] = True
        weights = valid.unsqueeze(-1).to(dtype=outputs.dtype)
        mean = (outputs * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        maximum = outputs.masked_fill(~valid.unsqueeze(-1), torch.finfo(outputs.dtype).min).max(dim=1).values
        return self.classifier(self.dropout(torch.cat([mean, maximum], dim=-1)))
