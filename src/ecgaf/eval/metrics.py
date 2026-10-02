"""Official PhysioNet/CinC 2017 score: mean F1 of Normal, AF, and Other."""

from __future__ import annotations

import numpy as np

CLASSES: tuple[str, ...] = ("N", "A", "O", "~")
SCORED_CLASSES: tuple[str, ...] = ("N", "A", "O")
CLASS_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(CLASSES)}
INDEX_TO_CLASS: dict[int, str] = {index: name for name, index in CLASS_TO_INDEX.items()}


def class_f1(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> float:
    """One-versus-rest F1. A class absent from both arrays scores 0."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    true_positive = int(np.sum((true == label) & (pred == label)))
    false_positive = int(np.sum((true != label) & (pred == label)))
    false_negative = int(np.sum((true == label) & (pred != label)))
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return (2 * true_positive) / denominator


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {label: class_f1(y_true, y_pred, label) for label in CLASSES}


def challenge_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean F1 of N, A, and O. Noise stays out of the average."""
    scores = [class_f1(y_true, y_pred, label) for label in SCORED_CLASSES]
    return float(np.mean(scores))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """Rows are true class, columns are predicted class, order N, A, O, ~."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for row, label_true in enumerate(CLASSES):
        for column, label_pred in enumerate(CLASSES):
            matrix[row, column] = int(np.sum((true == label_true) & (pred == label_pred)))
    return matrix.tolist()


def indices_to_labels(indices: np.ndarray) -> np.ndarray:
    mapper = np.array(CLASSES)
    return mapper[np.asarray(indices, dtype=int)]
