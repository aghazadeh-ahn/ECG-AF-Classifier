"""Band-pass filtering."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(signal: np.ndarray, sampling_rate: int, low_hz: float, high_hz: float, order: int) -> np.ndarray:
    nyquist = sampling_rate / 2
    high = min(high_hz, nyquist * 0.95)
    low = max(low_hz, 0.05)
    if high <= low:
        return np.asarray(signal, dtype=np.float32)
    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    padlen = 3 * (2 * sos.shape[0])
    if signal.size <= padlen:
        return np.asarray(signal, dtype=np.float32)
    filtered = sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))
    return filtered.astype(np.float32)
