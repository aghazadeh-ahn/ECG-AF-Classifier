"""Keep the whole recording, padded or center-cropped to a fixed length."""

from __future__ import annotations

import numpy as np


def fix_length(
    signal: np.ndarray,
    sampling_rate: int,
    seconds: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signal and a mask. In mask mode, padding is 0 on the mask.

    hard_crop uses the same crop and pad geometry, but the mask is all ones so
    the model treats padding as if it were signal.
    """
    target = int(round(seconds * sampling_rate))
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == target:
        return signal.copy(), np.ones(target, dtype=np.float32)
    if signal.size > target:
        start = (signal.size - target) // 2
        cropped = signal[start : start + target].copy()
        return cropped, np.ones(target, dtype=np.float32)
    start = (target - signal.size) // 2
    padded = np.zeros(target, dtype=np.float32)
    padded[start : start + signal.size] = signal
    if mode == "hard_crop":
        mask = np.ones(target, dtype=np.float32)
    else:
        mask = np.zeros(target, dtype=np.float32)
        mask[start : start + signal.size] = 1.0
    return padded, mask
