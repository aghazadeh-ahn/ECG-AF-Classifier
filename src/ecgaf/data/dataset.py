"""Torch dataset over a cached, fixed-length preprocessing of every record."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ecgaf.eval.metrics import CLASS_TO_INDEX
from ecgaf.preprocess.augment import augment_fixed


class ECGDataset(Dataset):
    def __init__(
        self,
        indices: list[int] | np.ndarray,
        labels: list[str],
        signals: np.ndarray,
        masks: np.ndarray,
        preprocess_cfg: dict,
        augment: bool,
        seed: int,
        sampling_rate: int = 300,
    ) -> None:
        self.indices = np.asarray(indices, dtype=int)
        self.labels = labels
        self.signals = signals
        self.masks = masks
        self.preprocess_cfg = preprocess_cfg
        self.augment = augment
        self.seed = seed
        self.sampling_rate = sampling_rate
        self.epoch = 0
        self.rr: np.ndarray | None = None
        self.rr_mean: np.ndarray | None = None
        self.rr_std: np.ndarray | None = None

    def set_rr(self, rr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        self.rr = rr
        self.rr_mean = mean.astype(np.float32)
        self.rr_std = std.astype(np.float32)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        global_index = int(self.indices[item])
        signal = np.array(self.signals[global_index], dtype=np.float32, copy=True)
        mask = np.array(self.masks[global_index], dtype=np.float32, copy=True)
        if self.augment and self.preprocess_cfg.get("augment", {}).get("enabled", False):
            rng = np.random.default_rng(self.seed + self.epoch * 100_003 + item)
            signal, mask = augment_fixed(
                signal,
                mask,
                self.preprocess_cfg["augment"],
                self.sampling_rate,
                rng,
            )
        label = CLASS_TO_INDEX[self.labels[global_index]]
        signal_tensor = torch.from_numpy(signal)
        mask_tensor = torch.from_numpy(mask)
        if self.rr is None:
            return signal_tensor, mask_tensor, label
        features = (self.rr[global_index] - self.rr_mean) / self.rr_std
        return signal_tensor, mask_tensor, torch.from_numpy(features.astype(np.float32)), label
