"""Compose preprocessing steps from a config mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecgaf.preprocess.filter import bandpass
from ecgaf.preprocess.invert import correct_inversion
from ecgaf.preprocess.length import fix_length
from ecgaf.preprocess.normalize import zscore

PIPELINE_VERSION = 1


@dataclass
class ProcessedSignal:
    signal: np.ndarray
    mask: np.ndarray
    sampling_rate: int
    inverted: bool


def preprocess_signal(
    signal: np.ndarray,
    sampling_rate: int,
    cfg: dict,
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> ProcessedSignal:
    del augment, rng  # augmentation is applied later on the cached fixed-length array
    waveform = np.asarray(signal, dtype=np.float32)
    band = cfg["bandpass"]
    if band["enabled"]:
        waveform = bandpass(
            waveform,
            sampling_rate,
            low_hz=float(band["low_hz"]),
            high_hz=float(band["high_hz"]),
            order=int(band["order"]),
        )
    waveform, inverted = correct_inversion(waveform, sampling_rate, enabled=bool(cfg["invert"]["enabled"]))
    if cfg["normalize"]["enabled"]:
        waveform = zscore(waveform)
    length_cfg = cfg["length"]
    if length_cfg["enabled"]:
        waveform, mask = fix_length(
            waveform,
            sampling_rate,
            seconds=float(length_cfg["seconds"]),
            mode=str(length_cfg["mode"]),
        )
    else:
        mask = np.ones(waveform.shape[0], dtype=np.float32)
    return ProcessedSignal(
        signal=np.asarray(waveform, dtype=np.float32),
        mask=np.asarray(mask, dtype=np.float32),
        sampling_rate=sampling_rate,
        inverted=inverted,
    )
