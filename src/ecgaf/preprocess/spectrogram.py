"""Log-magnitude spectrogram of an already preprocessed recording."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal


def log_spectrogram(
    waveform: np.ndarray,
    mask: np.ndarray,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log1p magnitude (frequency, time) and a time mask for valid frames."""
    waveform = np.asarray(waveform, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    _frequencies, _times, spectrum = scipy_signal.stft(
        waveform,
        fs=1.0,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False,
    )
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    frame_mask = _frame_mask(mask, magnitude.shape[1], n_fft, hop_length)
    valid = frame_mask >= 0.5
    if int(valid.sum()) > 0:
        deviation = float(magnitude[:, valid].std())
        if deviation > 1e-6:
            magnitude = magnitude / np.float32(deviation)
    return magnitude, frame_mask


def spectrogram_batch(
    waveforms: np.ndarray,
    masks: np.ndarray,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a cached waveform matrix into one spectrogram per record."""
    sample, sample_mask = log_spectrogram(waveforms[0], masks[0], n_fft, hop_length)
    spectra = np.zeros((len(waveforms), *sample.shape), dtype=np.float32)
    frame_masks = np.zeros((len(masks), sample_mask.shape[0]), dtype=np.float32)
    spectra[0] = sample
    frame_masks[0] = sample_mask
    for index in range(1, len(waveforms)):
        spectra[index], frame_masks[index] = log_spectrogram(
            waveforms[index], masks[index], n_fft, hop_length
        )
        if index % 1000 == 0:
            print(f"spectrogram {index}/{len(waveforms)}", flush=True)
    return spectra, frame_masks


def apply_spec_augment(
    spectrum: np.ndarray,
    frame_mask: np.ndarray,
    augment_cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero random frequency bands and time spans. Time spans leave the mask."""
    spectrum = np.array(spectrum, dtype=np.float32, copy=True)
    frame_mask = np.array(frame_mask, dtype=np.float32, copy=True)
    _mask_axis(spectrum, frame_mask, int(augment_cfg.get("freq_masks", 0)), int(augment_cfg.get("max_freq_bins", 0)), axis=0, rng=rng, drop_mask=False)
    _mask_axis(spectrum, frame_mask, int(augment_cfg.get("time_masks", 0)), int(augment_cfg.get("max_time_frames", 0)), axis=1, rng=rng, drop_mask=True)
    return spectrum, frame_mask


def _mask_axis(
    spectrum: np.ndarray,
    frame_mask: np.ndarray,
    masks: int,
    max_width: int,
    axis: int,
    rng: np.random.Generator,
    drop_mask: bool,
) -> None:
    if masks <= 0 or max_width <= 0:
        return
    length = spectrum.shape[axis]
    width_limit = min(max_width, max(length // 2, 1))
    for _ in range(masks):
        width = int(rng.integers(0, width_limit + 1))
        if width < 1 or width >= length:
            continue
        start = int(rng.integers(0, length - width + 1))
        if axis == 0:
            spectrum[start : start + width, :] = 0.0
        else:
            spectrum[:, start : start + width] = 0.0
            if drop_mask:
                frame_mask[start : start + width] = 0.0


def _frame_mask(mask: np.ndarray, n_frames: int, n_fft: int, hop_length: int) -> np.ndarray:
    frame_mask = np.zeros(n_frames, dtype=np.float32)
    for index in range(n_frames):
        start = index * hop_length
        end = start + n_fft
        if end > mask.size:
            break
        if float(mask[start:end].mean()) >= 0.5:
            frame_mask[index] = 1.0
    return frame_mask
