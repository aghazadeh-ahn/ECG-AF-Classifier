"""Class scales chosen on out-of-fold probabilities, then applied unchanged."""

from __future__ import annotations

import itertools

import numpy as np

from ecgaf.eval.metrics import CLASSES, challenge_f1, indices_to_labels, per_class_f1


def predict_with_scales(probabilities: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Divide each class probability by its scale, then take the winning class."""
    scaled = np.asarray(probabilities, dtype=np.float64) / np.asarray(scales, dtype=np.float64)
    return indices_to_labels(scaled.argmax(axis=1))


def choose_class_scales(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    grid: tuple[float, ...] = (0.7, 0.85, 1.0, 1.15, 1.35, 1.6),
) -> dict:
    """Search scales for AF, Other, and noise. Normal stays at 1. Test labels are not used."""
    best_scales = np.ones(len(CLASSES), dtype=np.float64)
    best_score = -1.0
    for other_scales in itertools.product(grid, repeat=3):
        scales = np.array([1.0, *other_scales], dtype=np.float64)
        score = challenge_f1(y_true, predict_with_scales(probabilities, scales))
        if score > best_score:
            best_score = score
            best_scales = scales
    return {
        "scales": {label: float(scale) for label, scale in zip(CLASSES, best_scales)},
        "challenge_f1": float(best_score),
        "per_class_f1": per_class_f1(y_true, predict_with_scales(probabilities, best_scales)),
    }
