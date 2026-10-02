"""Register architectures on import."""

from ecgaf.models.cnn1d import SmallCNN
from ecgaf.models.dilated_resnet import DilatedResNet
from ecgaf.models.registry import build_model, count_parameters

__all__ = ["SmallCNN", "DilatedResNet", "build_model", "count_parameters"]
