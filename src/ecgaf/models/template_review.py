"""A small network that sees the usual beat and the odd beat, then chooses Normal or Other."""

from __future__ import annotations

import torch
from torch import nn

from ecgaf.models.registry import register


@register("template_review")
def build_template_review(model_cfg: dict) -> nn.Module:
    return TemplateReview(dropout=float(model_cfg.get("dropout", 0.2)))


class TemplateReview(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, 2)

    def forward(self, templates: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder(templates).squeeze(-1)
        return self.classifier(self.dropout(pooled))
