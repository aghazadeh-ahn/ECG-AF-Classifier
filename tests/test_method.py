"""Scores, splits, preprocessing, and model shapes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ecgaf.config import load_experiment, project_root
from ecgaf.eval.metrics import CLASS_TO_INDEX, challenge_f1, class_f1
from ecgaf.eval.selective import choose_threshold, selective_scores
from ecgaf.eval.splits import assert_disjoint, make_splits
from ecgaf.models.registry import build_model
from ecgaf.preprocess.invert import correct_inversion
from ecgaf.preprocess.length import fix_length
from ecgaf.preprocess.pipeline import preprocess_signal
from ecgaf.train.weights import class_weights


def _preprocess_cfg(**overrides) -> dict:
    cfg = {
        "bandpass": {"enabled": False, "low_hz": 0.5, "high_hz": 40.0, "order": 4},
        "invert": {"enabled": True},
        "normalize": {"enabled": False},
        "length": {"enabled": True, "seconds": 60, "mode": "mask"},
        "augment": {"enabled": False},
    }
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def test_challenge_f1_matches_the_three_class_mean():
    y_true = np.array(["N", "N", "A", "O"])
    y_pred = np.array(["N", "A", "A", "O"])
    assert class_f1(y_true, y_pred, "N") == 2 / 3
    assert class_f1(y_true, y_pred, "A") == 2 / 3
    assert class_f1(y_true, y_pred, "O") == 1
    assert challenge_f1(y_true, y_pred) == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)
    assert class_f1(y_true, y_pred, "~") == 0


def test_noise_does_not_enter_the_challenge_average():
    y_true = np.array(["N", "~"])
    y_pred = np.array(["N", "N"])
    three_class = challenge_f1(y_true, y_pred)
    four_class = np.mean([class_f1(y_true, y_pred, label) for label in ("N", "A", "O", "~")])
    assert three_class != four_class


def test_locked_split_keeps_every_record_on_one_side():
    labels = ["N"] * 50 + ["A"] * 20 + ["O"] * 20 + ["~"] * 10
    splits = make_splits(labels, seed=42, test_size=0.2, n_folds=5)
    assert_disjoint(splits)
    assert set(splits["test"]).isdisjoint(splits["trainval"])


def test_short_recording_is_padded_and_masked():
    signal = np.ones(3000, dtype=np.float32)
    padded, mask = fix_length(signal, sampling_rate=300, seconds=60, mode="mask")
    assert padded.shape == (18000,)
    assert int(mask.sum()) == 3000
    hard, hard_mask = fix_length(signal, sampling_rate=300, seconds=60, mode="hard_crop")
    assert int(hard_mask.sum()) == 18000
    assert np.array_equal(hard, padded)


def test_long_recording_is_center_cropped():
    signal = np.arange(20000, dtype=np.float32)
    cropped, mask = fix_length(signal, sampling_rate=300, seconds=60, mode="mask")
    assert cropped.shape == (18000,)
    assert mask.sum() == 18000
    assert cropped[0] == 1000


def test_negative_qrs_is_flipped_and_positive_qrs_is_kept():
    sampling_rate = 300
    negative = np.zeros(8 * sampling_rate, dtype=np.float32)
    positive = np.zeros_like(negative)
    for beat in range(1, 7):
        center = beat * sampling_rate
        negative[center - 1 : center + 2] = np.array([-1.0, -2.0, -1.0], dtype=np.float32)
        positive[center - 1 : center + 2] = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    flipped, inverted = correct_inversion(negative, sampling_rate, enabled=True)
    kept, not_inverted = correct_inversion(positive, sampling_rate, enabled=True)
    unchanged, disabled = correct_inversion(negative, sampling_rate, enabled=False)
    assert inverted and flipped[sampling_rate] > 0
    assert not not_inverted and kept[sampling_rate] > 0
    assert not disabled and unchanged[sampling_rate] < 0


def test_pipeline_mask_follows_the_length_mode():
    signal = np.ones(3000, dtype=np.float32)
    masked = preprocess_signal(signal, 300, _preprocess_cfg(), augment=False)
    forced = preprocess_signal(
        signal,
        300,
        _preprocess_cfg(length={"enabled": True, "seconds": 60, "mode": "hard_crop"}),
        augment=False,
    )
    assert int(masked.mask.sum()) == 3000
    assert int(forced.mask.sum()) == 18000


def test_rare_class_gets_a_larger_loss_weight():
    labels = ["N"] * 90 + ["A"] * 10
    weights = class_weights(labels, range(100))
    assert weights[CLASS_TO_INDEX["A"]] > weights[CLASS_TO_INDEX["N"]]


def test_models_read_a_masked_minute():
    signal = torch.randn(2, 18000)
    mask = torch.ones(2, 18000)
    mask[:, 15000:] = 0
    cnn = build_model({"name": "cnn1d", "channels": [8, 16], "dropout": 0.0})
    residual = build_model(
        {"name": "dilated_resnet", "channels": [8, 8], "dilations": [1, 4], "dropout": 0.0}
    )
    plain = build_model(
        {"name": "dilated_resnet", "channels": [8, 8], "dilations": [1, 1], "dropout": 0.0}
    )
    assert cnn(signal, mask).shape == (2, 4)
    assert residual(signal, mask).shape == (2, 4)
    assert residual.dilations == [1, 4]
    assert plain.dilations == [1, 1]


def test_threshold_is_chosen_from_the_arrays_it_is_given():
    y_true = np.array(["N", "N", "A", "O", "O"])
    y_pred = np.array(["N", "A", "A", "O", "N"])
    confidence = np.array([0.95, 0.2, 0.9, 0.85, 0.4])
    chosen = choose_threshold(
        y_true,
        y_pred,
        confidence,
        thresholds=[0.5, 0.8],
        min_coverage=0.4,
    )
    assert chosen["chosen"]["threshold"] in (0.5, 0.8)
    accepted = selective_scores(y_true, y_pred, confidence, 0.8)
    assert accepted["n_accepted"] == 3
    assert accepted["challenge_f1_all"] == challenge_f1(y_true, y_pred)


def test_experiment_files_keep_the_same_split_and_encode_ablations():
    root = project_root()
    proposed = load_experiment(root / "configs" / "experiments" / "03_dilated_resnet.yaml")
    no_invert = load_experiment(root / "configs" / "experiments" / "03a_no_inversion.yaml")
    no_dilation = load_experiment(root / "configs" / "experiments" / "03b_no_dilation.yaml")
    hard_crop = load_experiment(root / "configs" / "experiments" / "03c_hard_crop.yaml")
    assert proposed["split"] == no_invert["split"] == no_dilation["split"] == hard_crop["split"]
    assert proposed["preprocess_cfg"]["invert"]["enabled"] is True
    assert no_invert["preprocess_cfg"]["invert"]["enabled"] is False
    assert no_dilation["model"]["dilations"] == [1, 1, 1, 1]
    assert hard_crop["preprocess_cfg"]["length"]["mode"] == "hard_crop"
    assert proposed["preprocess_cfg"]["length"]["mode"] == "mask"


def test_real_record_loads_when_the_challenge_files_are_present():
    data_root = project_root() / "data" / "raw"
    reference = data_root / "REFERENCE.csv"
    if not reference.exists():
        return
    from ecgaf.data.records import load_records, load_signal

    record = load_records(data_root)[0]
    signal, sampling_rate = load_signal(record)
    assert sampling_rate == 300
    assert signal.ndim == 1 and signal.size > 0
    processed = preprocess_signal(signal, sampling_rate, load_experiment(
        project_root() / "configs" / "experiments" / "03_dilated_resnet.yaml"
    )["preprocess_cfg"])
    assert processed.signal.shape == processed.mask.shape == (18000,)
    assert Path(record.mat_path).exists()
