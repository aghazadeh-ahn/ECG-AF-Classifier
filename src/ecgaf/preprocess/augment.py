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
    max_shift = int(augment_cfg.get("max_shift_seconds", 0.0) * sampling_rate)
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
    scale = float(rng.uniform(augment_cfg.get("scale_min", 1.0), augment_cfg.get("scale_max", 1.0)))
    noise_std = float(augment_cfg.get("noise_std", 0.0))
    noise = rng.normal(0.0, noise_std, size=shifted_signal.shape).astype(np.float32)
    augmented = shifted_signal * np.float32(scale) + noise * shifted_mask
    return augmented.astype(np.float32), shifted_mask.astype(np.float32)
