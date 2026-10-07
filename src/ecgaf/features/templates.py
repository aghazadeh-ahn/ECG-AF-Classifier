"""Two beat shapes: the usual beat, and the beat least like it."""

from __future__ import annotations

import numpy as np

from ecgaf.features.beats import aligned_beats


def two_templates(
    waveform: np.ndarray,
    mask: np.ndarray,
    sampling_rate: int,
    beat_samples: int,
    pre_samples: int,
) -> np.ndarray:
    """Return (2, beat_samples). Row 0 is the median beat. Row 1 is the farthest beat.

    A record with one beat has no second shape, so row 1 stays zero.
    """
    templates = np.zeros((2, beat_samples), dtype=np.float32)
    beats = aligned_beats(waveform, mask, sampling_rate, beat_samples, pre_samples)
    if beats.shape[0] == 0:
        return templates
    median = np.median(beats, axis=0).astype(np.float32)
    templates[0] = median
    if beats.shape[0] == 1:
        return templates
    distance = np.linalg.norm(beats - median, axis=1)
    templates[1] = beats[int(np.argmax(distance))]
    return templates


def two_template_batch(
    waveforms: np.ndarray,
    masks: np.ndarray,
    sampling_rate: int,
    beat_samples: int,
    pre_samples: int,
) -> np.ndarray:
    templates = np.zeros((len(waveforms), 2, beat_samples), dtype=np.float32)
    for index in range(len(waveforms)):
        templates[index] = two_templates(
            waveforms[index],
            masks[index],
            sampling_rate,
            beat_samples,
            pre_samples,
        )
        if index % 1000 == 0:
            print(f"templates {index}/{len(waveforms)}", flush=True)
    return templates
