"""Per-recording z-score."""

from __future__ import annotations

import numpy as np


def zscore(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    center = float(signal.mean())
    scale = float(signal.std())
    if scale < 1e-8:
        return signal - np.float32(center)
    return (signal - np.float32(center)) / np.float32(scale)
