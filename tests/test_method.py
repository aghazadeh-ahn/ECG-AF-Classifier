"""Scores, splits, preprocessing, and model shapes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ecgaf.config import load_experiment, project_root
from ecgaf.eval.decision import choose_class_scales, predict_with_scales
from ecgaf.eval.metrics import CLASS_TO_INDEX, challenge_f1, class_f1
from ecgaf.eval.selective import choose_threshold, selective_scores
from ecgaf.eval.splits import assert_disjoint, make_splits
from ecgaf.models.registry import build_model
from ecgaf.data.windows import random_crop, sliding_windows
from ecgaf.preprocess.augment import augment_fixed
from ecgaf.eval.review import review_normal_calls
from ecgaf.features.beats import beat_sequence
from ecgaf.features.templates import two_templates
from ecgaf.models.beat_lstm import BeatLSTM
from ecgaf.preprocess.spectrogram import apply_spec_augment, log_spectrogram
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
    wide = build_model(
        {
            "name": "dilated_resnet",
            "channels": [32, 64, 128, 256, 256],
            "dilations": [1, 2, 4, 8, 16],
            "dropout": 0.0,
        }
    )
    assert cnn(signal, mask).shape == (2, 4)
    assert residual(signal, mask).shape == (2, 4)
    assert residual.dilations == [1, 4]
    assert plain.dilations == [1, 1]
    assert wide(signal, mask).shape == (2, 4)
    assert wide.dilations == [1, 2, 4, 8, 16]
    attended = build_model(
        {
            "name": "dilated_resnet",
            "channels": [8, 8],
            "dilations": [1, 4],
            "dropout": 0.0,
            "pooling": "attention",
        }
    )
    assert attended(signal, mask).shape == (2, 4)
    assert attended.pooling_name == "attention"
    assert attended.attention is not None
    fused = build_model(
        {
            "name": "dilated_resnet_rr",
            "channels": [8, 8],
            "dilations": [1, 2],
            "dropout": 0.0,
            "rr_dim": 16,
        }
    )
    rhythm = torch.randn(2, 16)
    recurrent = build_model(
        {"name": "dilated_resnet_gru", "channels": [8, 8], "dilations": [1, 2], "dropout": 0.0, "gru_hidden": 8}
    )
    transformer = build_model(
        {
            "name": "dilated_resnet_transformer",
            "channels": [8, 8],
            "dilations": [1, 2],
            "dropout": 0.0,
            "transformer_heads": 2,
            "transformer_layers": 1,
            "transformer_ff": 16,
        }
    )
    assert fused(signal, mask, rhythm).shape == (2, 4)
    assert recurrent(signal, mask).shape == (2, 4)
    assert transformer(signal, mask).shape == (2, 4)
    spectrum = torch.randn(2, 65, 80)
    spectrum_mask = torch.ones(2, 80)
    spectrum_mask[:, 60:] = 0
    spectral = build_model(
        {"name": "spec_resnet", "channels": [8, 8], "dilations": [1, 2], "dropout": 0.0}
    )
    assert spectral(spectrum, spectrum_mask).shape == (2, 4)
    assert spectral.dilations == [1, 2]


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
    longer = load_experiment(root / "configs" / "experiments" / "10_long_cosine.yaml")
    assert longer["split"] == proposed["split"]
    assert longer["preprocess_cfg"] == proposed["preprocess_cfg"]
    assert longer["model"] == proposed["model"]
    assert longer["train"]["epochs"] == 100
    assert longer["train"]["scheduler"] == "cosine"
    assert longer["train"]["score_test"] == "fold_ensemble"
    strong = load_experiment(root / "configs" / "experiments" / "11_strong_augment.yaml")
    assert strong["split"] == longer["split"]
    assert strong["model"] == longer["model"]
    assert strong["train"] == longer["train"]
    assert strong["preprocess_cfg"]["bandpass"] == longer["preprocess_cfg"]["bandpass"]
    assert strong["preprocess_cfg"]["augment"]["scale_min"] == 0.7
    assert strong["preprocess_cfg"]["augment"]["warp_min"] == 0.9
    assert strong["preprocess_cfg"]["augment"]["cutout_seconds"] == 1.5
    windows = load_experiment(root / "configs" / "experiments" / "12_windows.yaml")
    assert windows["split"] == strong["split"]
    assert windows["model"] == strong["model"]
    assert windows["preprocess_cfg"] == strong["preprocess_cfg"]
    assert windows["train"]["scheduler"] == "cosine"
    assert windows["train"]["window"]["train_min_seconds"] == 10
    assert windows["train"]["window"]["eval_hop_seconds"] == 10
    spectral = load_experiment(root / "configs" / "experiments" / "14_log_spectrogram.yaml")
    assert spectral["split"] == proposed["split"]
    assert spectral["preprocess_cfg"]["bandpass"] == proposed["preprocess_cfg"]["bandpass"]
    assert spectral["preprocess_cfg"]["invert"]["enabled"] is True
    assert spectral["preprocess_cfg"]["augment"]["enabled"] is False
    assert spectral["representation"]["type"] == "log_spectrogram"
    assert spectral["model"]["name"] == "spec_resnet"
    assert spectral["train"]["score_test"] == "fold_ensemble"
    sequence = load_experiment(root / "configs" / "experiments" / "15_beat_lstm.yaml")
    assert sequence["split"] == proposed["split"]
    assert sequence["preprocess_cfg"]["bandpass"] == proposed["preprocess_cfg"]["bandpass"]
    assert sequence["preprocess_cfg"]["invert"]["enabled"] is True
    assert sequence["preprocess_cfg"]["augment"]["enabled"] is False
    assert sequence["representation"]["type"] == "beat_sequence"
    assert sequence["model"]["name"] == "beat_lstm"
    assert sequence["train"]["score_test"] == "fold_ensemble"
    pooled = load_experiment(root / "configs" / "experiments" / "17_mean_max_pool.yaml")
    best = load_experiment(root / "configs" / "experiments" / "11_strong_augment.yaml")
    assert pooled["split"] == best["split"]
    assert pooled["preprocess_cfg"] == best["preprocess_cfg"]
    assert pooled["train"] == best["train"]
    assert {key: value for key, value in pooled["model"].items() if key != "pooling"} == best["model"]
    assert pooled["model"]["pooling"] == "mean_max"
    review = load_experiment(root / "configs" / "experiments" / "16_normal_review.yaml")
    winner = load_experiment(root / "configs" / "experiments" / "11_strong_augment.yaml")
    assert review["split"] == winner["split"]
    assert review["preprocess_cfg"]["bandpass"] == winner["preprocess_cfg"]["bandpass"]
    assert review["preprocess_cfg"]["invert"]["enabled"] is True
    assert review["source_run"] == "runs/11_strong_augment"
    assert review["representation"]["type"] == "two_templates"
    assert review["model"]["name"] == "template_review"
    assert review["train"]["score_test"] == "fold_ensemble"


def test_log_spectrogram_hides_padding_and_spec_augment_keeps_shape():
    waveform = np.zeros(1800, dtype=np.float32)
    waveform[600:1200] = np.sin(np.linspace(0, 40, 600)).astype(np.float32)
    mask = np.zeros(1800, dtype=np.float32)
    mask[600:1200] = 1
    spectrum, frame_mask = log_spectrogram(waveform, mask, n_fft=128, hop_length=64)
    assert spectrum.ndim == 2
    assert spectrum.shape[0] == 65
    assert frame_mask.shape == (spectrum.shape[1],)
    assert frame_mask.sum() < frame_mask.size
    assert frame_mask.sum() > 0
    augmented, augmented_mask = apply_spec_augment(
        spectrum,
        frame_mask,
        {"freq_masks": 1, "max_freq_bins": 4, "time_masks": 1, "max_time_frames": 2},
        np.random.default_rng(0),
    )
    assert augmented.shape == spectrum.shape
    assert augmented_mask.shape == frame_mask.shape
    assert augmented_mask.sum() <= frame_mask.sum()


def test_beat_sequence_keeps_each_peak_and_the_lstm_uses_mean_and_max():
    sampling_rate = 300
    waveform = np.zeros(3000, dtype=np.float32)
    peaks = np.arange(300, 2700, 300)
    waveform[peaks] = 8.0
    mask = np.ones(3000, dtype=np.float32)
    beats, beat_mask = beat_sequence(waveform, mask, sampling_rate, beat_samples=180, pre_samples=75, max_beats=16)
    assert int(beat_mask.sum()) == peaks.size
    assert beats.shape == (16, 182)
    assert np.allclose(beats[0, -1], 1.0)
    assert beats[0, -2] == 0.0
    model = BeatLSTM(beat_length=180, hidden=8, dropout=0.0)
    logits = model(torch.from_numpy(beats).unsqueeze(0), torch.from_numpy(beat_mask).unsqueeze(0))
    assert logits.shape == (1, 4)


def test_odd_beat_becomes_the_second_template_and_only_normal_calls_can_change():
    sampling_rate = 300
    waveform = np.zeros(3000, dtype=np.float32)
    peaks = np.arange(300, 2700, 300)
    waveform[peaks] = 8.0
    odd = int(peaks[3])
    waveform[odd - 12 : odd + 12] = 4.0
    mask = np.ones(3000, dtype=np.float32)
    templates = two_templates(waveform, mask, sampling_rate, beat_samples=180, pre_samples=75)
    assert templates.shape == (2, 180)
    beats = two_templates(waveform, mask, sampling_rate, beat_samples=180, pre_samples=75)
    distance = np.linalg.norm(beats[1] - beats[0])
    assert distance > 1.0
    base = np.array([0, 1, 2, 3, 0])
    revised = review_normal_calls(base, np.array([True, True, True, True, False]))
    assert revised.tolist() == [2, 1, 2, 3, 0]


def test_mean_max_pooling_sees_one_strong_step_that_the_mean_dilutes():
    from ecgaf.models.pooling import masked_global_average, masked_global_max

    values = torch.zeros(1, 2, 100)
    values[0, 0, 10] = 5.0
    mask = torch.ones(1, 100)
    mask[0, 50:] = 0
    assert float(masked_global_average(values, mask)[0, 0]) == pytest.approx(0.1)
    assert float(masked_global_max(values, mask)[0, 0]) == 5.0
    padded_peak = torch.zeros(1, 1, 100)
    padded_peak[0, 0, 80] = 9.0
    assert float(masked_global_max(padded_peak, mask)[0, 0]) == 0.0
    model = build_model(
        {"name": "dilated_resnet", "channels": [8, 8], "dilations": [1, 2], "dropout": 0.0, "pooling": "mean_max"}
    )
    assert model.classifier.in_features == 16
    assert model(torch.randn(2, 3000), torch.ones(2, 3000)).shape == (2, 4)


def test_random_crop_stays_inside_one_record_and_windows_cover_it():
    signal = np.arange(18_000, dtype=np.float32)
    mask = np.ones(18_000, dtype=np.float32)
    crop_cfg = {"train_min_seconds": 10, "train_max_seconds": 30}
    cropped, cropped_mask = random_crop(signal, mask, crop_cfg, 300, np.random.default_rng(0))
    assert cropped.shape == cropped_mask.shape == (9_000,)
    assert 3_000 <= int(cropped_mask.sum()) <= 9_000
    assert np.all(cropped[cropped_mask < 0.5] == 0)
    views = sliding_windows(signal, mask, window=9_000, hop=3_000)
    assert len(views) == 4
    assert views[-1][0][-1] == signal[-1]
    short = np.ones(2_700, dtype=np.float32)
    short_mask = np.ones(2_700, dtype=np.float32)
    short_views = sliding_windows(short, short_mask, window=9_000, hop=3_000)
    assert len(short_views) == 1
    assert int(short_views[0][1].sum()) == 2_700


def test_class_scales_are_chosen_without_looking_at_a_held_out_row():
    probabilities = np.array(
        [
            [0.42, 0.40, 0.15, 0.03],
            [0.38, 0.36, 0.20, 0.06],
            [0.80, 0.10, 0.08, 0.02],
        ],
        dtype=np.float64,
    )
    y_true = np.array(["A", "A", "N"])
    chosen = choose_class_scales(y_true, probabilities, grid=(0.7, 1.0))
    assert chosen["scales"]["N"] == 1.0
    assert chosen["scales"]["A"] == 0.7
    tuned = predict_with_scales(probabilities, np.array([1.0, chosen["scales"]["A"], chosen["scales"]["O"], chosen["scales"]["~"]]))
    assert challenge_f1(y_true, tuned) >= challenge_f1(y_true, predict_with_scales(probabilities, np.ones(4)))


def test_strong_augment_preserves_shape_and_drops_padding():
    signal = np.zeros(1800, dtype=np.float32)
    signal[600:1200] = np.linspace(-1.0, 1.0, 600, dtype=np.float32)
    mask = np.zeros(1800, dtype=np.float32)
    mask[600:1200] = 1.0
    plain, plain_mask = augment_fixed(
        signal,
        mask,
        {"max_shift_seconds": 0.0, "scale_min": 2.0, "scale_max": 2.0, "noise_std": 0.0},
        300,
        np.random.default_rng(0),
    )
    assert np.allclose(plain, signal * 2)
    assert np.array_equal(plain_mask, mask)
    augmented, augmented_mask = augment_fixed(
        signal,
        mask,
        {
            "max_shift_seconds": 0.0,
            "scale_min": 1.0,
            "scale_max": 1.0,
            "noise_std": 0.0,
            "warp_min": 0.9,
            "warp_max": 0.9,
            "wander_amp": 0.2,
            "wander_hz_min": 0.2,
            "wander_hz_max": 0.2,
            "cutout_seconds": 0.5,
        },
        300,
        np.random.default_rng(1),
    )
    assert augmented.shape == augmented_mask.shape == signal.shape
    assert np.all(augmented[augmented_mask < 0.5] == 0)
    assert augmented_mask.sum() < mask.sum()


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
