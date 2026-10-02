"""Read the public challenge records and the final REFERENCE.csv labels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io

_GAIN_PATTERN = re.compile(r"([0-9]*\.?[0-9]+)\s*/\s*mV", re.IGNORECASE)


@dataclass(frozen=True)
class Record:
    record_id: str
    mat_path: Path
    hea_path: Path
    label: str


def load_records(data_root: Path) -> list[Record]:
    reference = data_root / "REFERENCE.csv"
    records: list[Record] = []
    for line in reference.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record_id, label = line.split(",")
        record_id = record_id.strip()
        label = label.strip()
        records.append(
            Record(
                record_id=record_id,
                mat_path=data_root / f"{record_id}.mat",
                hea_path=data_root / f"{record_id}.hea",
                label=label,
            )
        )
    if not records:
        raise ValueError(f"No records found in {reference}")
    return records


def read_sampling_rate_and_gain(hea_path: Path) -> tuple[int, float]:
    lines = hea_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split()
    sampling_rate = int(float(header[2]))
    gain = 1000.0
    if len(lines) > 1:
        match = _GAIN_PATTERN.search(lines[1])
        if match:
            gain = float(match.group(1))
    if gain == 0:
        gain = 1000.0
    return sampling_rate, gain


def load_signal(record: Record) -> tuple[np.ndarray, int]:
    """Return the recording in millivolts and its sampling rate."""
    if not record.mat_path.exists():
        raise FileNotFoundError(record.mat_path)
    sampling_rate, gain = read_sampling_rate_and_gain(record.hea_path)
    mat = scipy.io.loadmat(record.mat_path)
    if "val" not in mat:
        raise KeyError(f"{record.mat_path} has no 'val' array")
    signal = np.asarray(mat["val"], dtype=np.float32).reshape(-1)
    signal = signal / np.float32(gain)
    return signal, sampling_rate
