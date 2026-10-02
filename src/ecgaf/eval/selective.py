"""Abstain when the winning class probability is below a threshold."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ecgaf.eval.metrics import challenge_f1


def selective_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Challenge F1 on every record, and again only on accepted records."""
    accepted = np.asarray(confidence) >= threshold
    coverage = float(np.mean(accepted)) if len(accepted) else 0.0
    if int(np.sum(accepted)) == 0:
        accepted_f1 = 0.0
    else:
        accepted_f1 = challenge_f1(y_true[accepted], y_pred[accepted])
    return {
        "threshold": float(threshold),
        "coverage": coverage,
        "challenge_f1_all": challenge_f1(y_true, y_pred),
        "challenge_f1_accepted": accepted_f1,
        "n_accepted": int(np.sum(accepted)),
        "n_total": int(len(y_true)),
    }


def choose_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    thresholds: list[float],
    min_coverage: float,
) -> dict:
    """Pick a threshold from out-of-fold scores. The locked test set is not used."""
    ranked = [
        selective_scores(y_true, y_pred, confidence, threshold) for threshold in thresholds
    ]
    eligible = [row for row in ranked if row["coverage"] >= min_coverage]
    pool = eligible or ranked
    best = max(pool, key=lambda row: row["challenge_f1_accepted"])
    return {"chosen": best, "sweep": ranked}


def load_probability_file(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}
