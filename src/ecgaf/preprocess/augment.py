"""Training-only changes that do not cross the train/test boundary."""

from __future__ import annotations

import numpy as np


def augment_fixed(
    signal: np.ndarray,
    mask: np.ndarray,
    augment_cfg: dict,
    sampling_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.array(signal, dtype=np.float32, copy=True)
    mask = np.array(mask, dtype=np.float32, copy=True)
    signal, mask = _shift(signal, mask, augment_cfg, sampling_rate, rng)
    signal, mask = _time_warp(signal, mask, augment_cfg, rng)
    signal = _baseline_wander(signal, mask, augment_cfg, sampling_rate, rng)
    signal = _scale_and_noise(signal, mask, augment_cfg, rng)
    return _cutout(signal, mask, augment_cfg, sampling_rate, rng)


def _shift(
    signal: np.ndarray,
    mask: np.ndarray,
    augment_cfg: dict,
    sampling_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    max_shift = int(float(augment_cfg.get("max_shift_seconds", 0.0)) * sampling_rate)
    shift = int(rng.integers(-max_shift, max_shift + 1)) if max_shift else 0
    shifted_signal = np.zeros_like(signal)
    shifted_mask = np.zeros_like(mask)
    if shift >= 0:
        shifted_signal[shift:] = signal[: signal.size - shift]
        shifted_mask[shift:] = mask[: mask.size - shift]
    else:
        amount = -shift
        shifted_signal[: signal.size - amount] = signal[amount:]
        shifted_mask[: mask.size - amount] = mask[amount:]
    return shifted_signal, shifted_mask


def _time_warp(
    signal: np.ndarray,
    mask: np.ndarray,
    augment_cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    warp_min = float(augment_cfg.get("warp_min", 1.0))
    warp_max = float(augment_cfg.get("warp_max", 1.0))
    if warp_min == 1.0 and warp_max == 1.0:
        return signal, mask
    bounds = _valid_bounds(mask)
    if bounds is None:
        return signal, mask
    start, end = bounds
    segment = signal[start:end]
    if segment.size < 4:
        return signal, mask
    factor = float(rng.uniform(warp_min, warp_max))
    new_length = int(round(segment.size * factor))
    new_length = min(max(new_length, 4), signal.size)
    source = np.arange(segment.size, dtype=np.float32)
    target = np.linspace(0, segment.size - 1, new_length, dtype=np.float32)
    resampled = np.interp(target, source, segment).astype(np.float32)
    warped = np.zeros_like(signal)
    warped_mask = np.zeros_like(mask)
    place = (signal.size - new_length) // 2
    warped[place : place + new_length] = resampled
    warped_mask[place : place + new_length] = 1.0
    return warped, warped_mask


def _baseline_wander(
    signal: np.ndarray,
    mask: np.ndarray,
    augment_cfg: dict,
    sampling_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    amplitude = float(augment_cfg.get("wander_amp", 0.0))
    if amplitude <= 0:
        return signal
    indices = np.flatnonzero(mask >= 0.5)
    if indices.size == 0:
        return signal
    low = float(augment_cfg.get("wander_hz_min", 0.05))
    high = float(augment_cfg.get("wander_hz_max", 0.5))
    frequency = float(rng.uniform(low, high)) if high > low else low
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    strength = float(rng.uniform(0.0, amplitude))
    time = indices.astype(np.float32) / float(sampling_rate)
    wave = np.sin((2.0 * np.pi * frequency * time) + phase).astype(np.float32)
    signal[indices] = signal[indices] + np.float32(strength) * wave
    return signal


def _scale_and_noise(
    signal: np.ndarray,
    mask: np.ndarray,
    augment_cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    scale = float(rng.uniform(augment_cfg.get("scale_min", 1.0), augment_cfg.get("scale_max", 1.0)))
    noise_std = float(augment_cfg.get("noise_std", 0.0))
    noise = rng.normal(0.0, noise_std, size=signal.shape).astype(np.float32)
    return signal * np.float32(scale) + noise * mask


def _cutout(
    signal: np.ndarray,
    mask: np.ndarray,
    augment_cfg: dict,
    sampling_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    seconds = float(augment_cfg.get("cutout_seconds", 0.0))
    if seconds <= 0:
        return signal, mask
    indices = np.flatnonzero(mask >= 0.5)
    width = int(round(seconds * sampling_rate))
    width = min(width, int(indices.size // 2))
    if indices.size < 2 or width < 1:
        return signal, mask
    start = int(rng.integers(0, indices.size - width + 1))
    chosen = indices[start : start + width]
    signal[chosen] = 0.0
    mask[chosen] = 0.0
    return signal, mask


def _valid_bounds(mask: np.ndarray) -> tuple[int, int] | None:
    indices = np.flatnonzero(mask >= 0.5)
    if indices.size == 0:
        return None
    return int(indices[0]), int(indices[-1]) + 1
