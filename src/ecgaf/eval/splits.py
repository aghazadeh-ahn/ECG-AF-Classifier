"""Locked stratified holdout plus folds on the remaining records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_splits(
    labels: list[str] | np.ndarray,
    seed: int,
    test_size: float,
    n_folds: int,
) -> dict:
    """Each record index appears in the locked test set or in trainval, never both.

    Fold indices are positions in the full record list. Validation folds partition
    trainval, so a record is never used to train the fold that scores it.
    """
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    trainval, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    folder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    trainval_labels = labels[trainval]
    for train_pos, val_pos in folder.split(trainval, trainval_labels):
        folds.append(
            {
                "train": trainval[train_pos].astype(int).tolist(),
                "val": trainval[val_pos].astype(int).tolist(),
            }
        )
    return {
        "seed": seed,
        "test_size": test_size,
        "n_folds": n_folds,
        "trainval": trainval.astype(int).tolist(),
        "test": test.astype(int).tolist(),
        "folds": folds,
    }


def assert_disjoint(splits: dict) -> None:
    test = set(splits["test"])
    trainval = set(splits["trainval"])
    if test & trainval:
        raise ValueError("A record is in both the locked test set and trainval")
    if len(test) + len(trainval) != len(test | trainval):
        raise ValueError("Duplicate indices inside a split")
    seen_val: set[int] = set()
    for fold in splits["folds"]:
        train = set(fold["train"])
        val = set(fold["val"])
        if train & val:
            raise ValueError("A record is in both train and validation of one fold")
        if train | val != trainval:
            raise ValueError("A fold does not partition trainval")
        if seen_val & val:
            raise ValueError("A record appears in more than one validation fold")
        seen_val |= val
    if seen_val != trainval:
        raise ValueError("Validation folds do not cover trainval")


def splits_path(processed_dir: Path, seed: int, test_size: float, n_folds: int) -> Path:
    test_token = str(test_size).replace(".", "")
    return processed_dir / f"splits_seed{seed}_test{test_token}_folds{n_folds}.json"


def load_or_create_splits(
    labels: list[str],
    processed_dir: Path,
    seed: int,
    test_size: float,
    n_folds: int,
) -> dict:
    path = splits_path(processed_dir, seed, test_size, n_folds)
    if path.exists():
        splits = json.loads(path.read_text(encoding="utf-8"))
        if len(splits["trainval"]) + len(splits["test"]) != len(labels):
            raise ValueError(f"Saved splits in {path} do not match the current records")
        assert_disjoint(splits)
        return splits
    splits = make_splits(labels, seed=seed, test_size=test_size, n_folds=n_folds)
    assert_disjoint(splits)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits), encoding="utf-8")
    return splits
