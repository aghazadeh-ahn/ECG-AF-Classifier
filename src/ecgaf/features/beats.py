"""One row per heartbeat: the wave around R, plus the RR interval on each side."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def apply_beat_noise(
    beats: np.ndarray,
    beat_length: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noisy = np.array(beats, dtype=np.float32, copy=True)
    noisy[:, :beat_length] += rng.normal(0.0, noise_std, size=(beats.shape[0], beat_length)).astype(np.float32)
    return noisy


def aligned_beats(
    waveform: np.ndarray,
    mask: np.ndarray,
    sampling_rate: int,
    beat_samples: int,
    pre_samples: int,
) -> np.ndarray:
    """One row per detected R peak, each row a fixed window starting before that peak."""
    peaks = _r_peaks(waveform, mask, sampling_rate)
    beats = np.zeros((peaks.size, beat_samples), dtype=np.float32)
    for index, peak in enumerate(peaks):
        start = int(peak) - pre_samples
        stop = start + beat_samples
        valid_start = max(start, 0)
        valid_stop = min(stop, waveform.size)
        destination = valid_start - start
        beats[index, destination : destination + (valid_stop - valid_start)] = waveform[valid_start:valid_stop]
    return beats


def beat_sequence(
    waveform: np.ndarray,
    mask: np.ndarray,
    sampling_rate: int,
    beat_samples: int,
    pre_samples: int,
    max_beats: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (max_beats, beat_samples + 2) and a mask over the beat axis."""
    features = np.zeros((max_beats, beat_samples + 2), dtype=np.float32)
    beat_mask = np.zeros(max_beats, dtype=np.float32)
    peaks = _r_peaks(waveform, mask, sampling_rate)
    if peaks.size == 0:
        return features, beat_mask
    intervals = np.diff(peaks).astype(np.float32) / float(sampling_rate)
    previous = np.zeros(peaks.size, dtype=np.float32)
    following = np.zeros(peaks.size, dtype=np.float32)
    if intervals.size:
        previous[1:] = intervals
        following[:-1] = intervals
    if peaks.size > max_beats:
        chosen = np.linspace(0, peaks.size - 1, max_beats).astype(int)
        peaks = peaks[chosen]
        previous = previous[chosen]
        following = following[chosen]
    for index, peak in enumerate(peaks):
        start = int(peak) - pre_samples
        stop = start + beat_samples
        valid_start = max(start, 0)
        valid_stop = min(stop, waveform.size)
        destination = valid_start - start
        features[index, destination : destination + (valid_stop - valid_start)] = waveform[valid_start:valid_stop]
        features[index, -2] = float(previous[index])
        features[index, -1] = float(following[index])
        beat_mask[index] = 1.0
    return features, beat_mask


def beat_sequence_batch(
    waveforms: np.ndarray,
    masks: np.ndarray,
    sampling_rate: int,
    beat_samples: int,
    pre_samples: int,
    max_beats: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequences = np.zeros((len(waveforms), max_beats, beat_samples + 2), dtype=np.float32)
    beat_masks = np.zeros((len(waveforms), max_beats), dtype=np.float32)
    for index in range(len(waveforms)):
        sequences[index], beat_masks[index] = beat_sequence(
            waveforms[index],
            masks[index],
            sampling_rate,
            beat_samples,
            pre_samples,
            max_beats,
        )
        if index % 1000 == 0:
            print(f"beats {index}/{len(waveforms)}", flush=True)
    return sequences, beat_masks


def _r_peaks(waveform: np.ndarray, mask: np.ndarray, sampling_rate: int) -> np.ndarray:
    valid = np.flatnonzero(np.asarray(mask) >= 0.5)
    if valid.size < sampling_rate:
        return np.empty(0, dtype=int)
    start = int(valid[0])
    end = int(valid[-1]) + 1
    segment = np.asarray(waveform[start:end], dtype=np.float32)
    deviation = float(np.std(segment))
    distance = max(1, int(0.25 * sampling_rate))
    prominence = max(0.5 * deviation, 1e-6)
    peaks, _ = find_peaks(segment, distance=distance, prominence=prominence)
    return peaks.astype(int) + start
