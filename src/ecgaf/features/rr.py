"""A short set of RR-interval statistics for the classical baseline."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

FEATURE_NAMES: tuple[str, ...] = (
    "duration_s",
    "n_beats",
    "rr_mean",
    "rr_std",
    "rr_min",
    "rr_max",
    "rr_median",
    "rmssd",
    "pnn50",
    "sd1",
    "sd2",
    "sd1_sd2",
    "cv",
    "hr_bpm",
    "rr_p25",
    "rr_p75",
)


def rr_features(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    duration = float(signal.size / sampling_rate) if sampling_rate else 0.0
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    features[0] = duration
    deviation = float(np.std(signal))
    distance = max(1, int(0.25 * sampling_rate))
    prominence = max(0.5 * deviation, 1e-6)
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
    features[1] = float(peaks.size)
    if peaks.size < 3:
        return features
    intervals = np.diff(peaks).astype(np.float64) / sampling_rate
    intervals = intervals[(intervals >= 0.3) & (intervals <= 2.0)]
    if intervals.size < 2:
        return features
    successive = np.diff(intervals)
    mean_rr = float(intervals.mean())
    std_rr = float(intervals.std())
    rmssd = float(np.sqrt(np.mean(successive**2))) if successive.size else 0.0
    sd1 = float(np.sqrt(0.5 * np.var(successive))) if successive.size else 0.0
    sd2_sq = 2 * float(np.var(intervals)) - 0.5 * (float(np.var(successive)) if successive.size else 0.0)
    sd2 = float(np.sqrt(max(sd2_sq, 0.0)))
    pnn50 = float(np.mean(np.abs(successive) > 0.05)) if successive.size else 0.0
    features[2:] = np.array(
        [
            mean_rr,
            std_rr,
            float(intervals.min()),
            float(intervals.max()),
            float(np.median(intervals)),
            rmssd,
            pnn50,
            sd1,
            sd2,
            sd1 / (sd2 + 1e-8),
            std_rr / (mean_rr + 1e-8),
            60.0 / (mean_rr + 1e-8),
            float(np.percentile(intervals, 25)),
            float(np.percentile(intervals, 75)),
        ],
        dtype=np.float32,
    )
    return features
