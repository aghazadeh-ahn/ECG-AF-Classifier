"""Inverse-frequency weights computed only on the records used for training."""

from __future__ import annotations

import numpy as np

from ecgaf.eval.metrics import CLASSES, CLASS_TO_INDEX


def class_weights(labels: list[str], indices: list[int] | np.ndarray) -> np.ndarray:
    picked = [labels[int(index)] for index in indices]
    counts = np.zeros(len(CLASSES), dtype=np.float64)
    for label in picked:
        counts[CLASS_TO_INDEX[label]] += 1
    weights = np.zeros(len(CLASSES), dtype=np.float32)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return weights
