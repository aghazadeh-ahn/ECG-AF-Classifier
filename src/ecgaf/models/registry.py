"""Model constructors, keyed by the name in an experiment file."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register(name: str) -> Callable:
    def decorator(builder: Callable) -> Callable:
        _REGISTRY[name] = builder
        return builder

    return decorator


def build_model(model_cfg: dict) -> nn.Module:
    import ecgaf.models.cnn1d  # noqa: F401
    import ecgaf.models.dilated_resnet  # noqa: F401

    name = model_cfg["name"]
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown model {name}. Known models: {known}")
    return _REGISTRY[name](model_cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
