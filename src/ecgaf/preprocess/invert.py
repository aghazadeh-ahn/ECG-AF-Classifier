"""Flip recordings whose QRS complexes point the wrong way."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def qrs_is_negative(signal: np.ndarray, sampling_rate: int) -> bool:
    """True when the large deflections are mostly negative, as in an inverted lead."""
    if signal.size < sampling_rate:
        return False
    deviation = float(np.std(signal))
    if deviation == 0:
        return False
    distance = max(1, int(0.3 * sampling_rate))
    peaks, _ = find_peaks(np.abs(signal), distance=distance, prominence=0.5 * deviation)
    if peaks.size < 3:
        return False
    return float(np.median(signal[peaks])) < 0


def correct_inversion(signal: np.ndarray, sampling_rate: int, enabled: bool) -> tuple[np.ndarray, bool]:
    if not enabled:
        return np.asarray(signal, dtype=np.float32), False
    signal = np.asarray(signal, dtype=np.float32)
    if qrs_is_negative(signal, sampling_rate):
        return -signal, True
    return signal, False
