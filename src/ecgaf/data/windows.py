"""Record-level crops for training and overlapping windows for scoring."""

from __future__ import annotations

import numpy as np


def random_crop(
    signal: np.ndarray,
    mask: np.ndarray,
    crop_cfg: dict,
    sampling_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Take one random span from the valid recording and pad it to a fixed length."""
    min_samples = int(round(float(crop_cfg["train_min_seconds"]) * sampling_rate))
    max_samples = int(round(float(crop_cfg["train_max_seconds"]) * sampling_rate))
    cropped = np.zeros(max_samples, dtype=np.float32)
    cropped_mask = np.zeros(max_samples, dtype=np.float32)
    bounds = _valid_bounds(mask)
    if bounds is None:
        return cropped, cropped_mask
    start, end = bounds
    valid = end - start
    duration = int(rng.integers(min_samples, max_samples + 1))
    duration = min(duration, valid, max_samples)
    duration = max(duration, 1)
    offset = start if valid <= duration else int(rng.integers(start, end - duration + 1))
    segment = np.asarray(signal[offset : offset + duration], dtype=np.float32)
    place = (max_samples - segment.size) // 2
    cropped[place : place + segment.size] = segment
    cropped_mask[place : place + segment.size] = 1.0
    return cropped, cropped_mask


def sliding_windows(
    signal: np.ndarray,
    mask: np.ndarray,
    window: int,
    hop: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cover the valid recording with fixed windows. A short recording is one padded window."""
    if window < 1 or hop < 1:
        raise ValueError("window and hop must be positive")
    bounds = _valid_bounds(mask)
    if bounds is None:
        return [(np.zeros(window, dtype=np.float32), np.zeros(window, dtype=np.float32))]
    start, end = bounds
    length = end - start
    if length <= window:
        padded = np.zeros(window, dtype=np.float32)
        padded_mask = np.zeros(window, dtype=np.float32)
        place = (window - length) // 2
        padded[place : place + length] = np.asarray(signal[start:end], dtype=np.float32)
        padded_mask[place : place + length] = 1.0
        return [(padded, padded_mask)]
    offsets = list(range(start, end - window + 1, hop))
    last = end - window
    if offsets[-1] != last:
        offsets.append(last)
    ones = np.ones(window, dtype=np.float32)
    return [
        (np.asarray(signal[offset : offset + window], dtype=np.float32).copy(), ones.copy())
        for offset in offsets
    ]


def _valid_bounds(mask: np.ndarray) -> tuple[int, int] | None:
    indices = np.flatnonzero(np.asarray(mask) >= 0.5)
    if indices.size == 0:
        return None
    return int(indices[0]), int(indices[-1]) + 1
