"""Disk cache of fixed-length signals so later epochs skip MATLAB reads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ecgaf.data.records import Record, load_signal
from ecgaf.preprocess.pipeline import PIPELINE_VERSION, ProcessedSignal, preprocess_signal

def preprocess_cache_key(preprocess_cfg: dict) -> str:
    payload = {"version": PIPELINE_VERSION, "preprocess": preprocess_cfg, "augment": False}
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def ensure_signal_cache(
    records: list[Record],
    preprocess_cfg: dict,
    processed_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return memmaps of signals, masks, and a boolean inversion flag per record."""
    key = preprocess_cache_key(preprocess_cfg)
    cache_dir = processed_dir / "cache" / key
    signals_path = cache_dir / "signals.npy"
    masks_path = cache_dir / "masks.npy"
    flags_path = cache_dir / "inverted.npy"
    meta_path = cache_dir / "meta.json"
    if meta_path.exists() and signals_path.exists() and masks_path.exists() and flags_path.exists():
        signals = np.load(signals_path, mmap_mode="r")
        masks = np.load(masks_path, mmap_mode="r")
        inverted = np.load(flags_path, mmap_mode="r")
        if len(signals) == len(records):
            return signals, masks, inverted

    sample_signal, sample_rate = load_signal(records[0])
    sample = preprocess_signal(sample_signal, sample_rate, preprocess_cfg, augment=False)
    if not preprocess_cfg["length"]["enabled"]:
        raise ValueError("Signal cache needs a fixed length. Enable preprocess.length.")
    target = sample.signal.shape[0]
    cache_dir.mkdir(parents=True, exist_ok=True)
    signals = np.lib.format.open_memmap(
        signals_path, mode="w+", dtype=np.float32, shape=(len(records), target)
    )
    masks = np.lib.format.open_memmap(
        masks_path, mode="w+", dtype=np.float32, shape=(len(records), target)
    )
    inverted = np.lib.format.open_memmap(
        flags_path, mode="w+", dtype=np.bool_, shape=(len(records),)
    )
    for index, record in enumerate(records):
        raw, sampling_rate = load_signal(record)
        processed: ProcessedSignal = preprocess_signal(
            raw, sampling_rate, preprocess_cfg, augment=False
        )
        signals[index] = processed.signal
        masks[index] = processed.mask
        inverted[index] = processed.inverted
        if index % 500 == 0:
            print(f"cached {index}/{len(records)}", flush=True)
    signals.flush()
    masks.flush()
    inverted.flush()
    fraction = float(np.mean(inverted))
    meta_path.write_text(
        json.dumps({"n": len(records), "length": target, "inverted_fraction": fraction}),
        encoding="utf-8",
    )
    print(f"lead inversion fraction {fraction:.3f}", flush=True)
    return signals, masks, inverted
