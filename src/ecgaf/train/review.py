"""Second decision for records the first model called Normal: usual beat versus odd beat."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecgaf.data.cache import ensure_signal_cache
from ecgaf.data.records import Record
from ecgaf.eval.metrics import (
    CLASS_TO_INDEX,
    challenge_f1,
    confusion_matrix,
    indices_to_labels,
    per_class_f1,
)
from ecgaf.eval.review import review_normal_calls
from ecgaf.features.templates import two_template_batch
from ecgaf.models.registry import build_model, count_parameters
from ecgaf.train.loop import _device, seed_everything


def run_template_review(experiment: dict, records: list[Record], splits: dict, run_dir: Path) -> dict:
    labels = [record.label for record in records]
    source = _load_source(experiment)
    base_oof = _prediction_indices(source["oof_pred"], len(records))
    if not np.all(source["oof_filled"][splits["trainval"]]):
        raise RuntimeError("Model 11 is missing out-of-fold predictions for trainval")
    templates = _templates(experiment, records)
    device = _device()
    train_cfg = experiment["train"]
    seed = int(experiment["split"]["seed"])
    started = time.perf_counter()
    fold_scores: list[float] = []
    base_fold_scores: list[float] = []
    best_epochs: list[int] = []
    fold_states: list[dict[str, torch.Tensor]] = []
    oof_pred = base_oof.copy()
    oof_filled = np.zeros(len(records), dtype=bool)

    for fold_id, fold in enumerate(splits["folds"]):
        print(f"fold {fold_id}", flush=True)
        model, best_epoch, score, base_score = _fit_fold(
            experiment,
            templates,
            labels,
            base_oof,
            fold,
            device,
            seed + fold_id,
        )
        fold_scores.append(score)
        base_fold_scores.append(base_score)
        best_epochs.append(best_epoch)
        fold_states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
        val_index = np.asarray(fold["val"], dtype=int)
        votes = _other_votes(model, templates[val_index], device, int(train_cfg["batch_size"]))
        oof_pred[val_index] = review_normal_calls(base_oof[val_index], votes)
        oof_filled[val_index] = True
        print(
            f"fold {fold_id} best epoch {best_epoch} "
            f"challenge F1 {score:.4f} base {base_score:.4f}",
            flush=True,
        )

    print("score locked test with the five fold models", flush=True)
    test_index = np.asarray(splits["test"], dtype=int)
    order = _test_order(source["test_index"], test_index)
    base_test = _prediction_indices(source["test_pred"], len(test_index))[order]
    accumulated = np.zeros((len(test_index), 2), dtype=np.float64)
    for state in fold_states:
        fold_model = build_model(experiment["model"])
        fold_model.load_state_dict(state)
        fold_model.to(device)
        accumulated += _probabilities(fold_model, templates[test_index], device, int(train_cfg["batch_size"]))
    mean_probability = accumulated / len(fold_states)
    test_pred = review_normal_calls(base_test, mean_probability[:, 1] >= mean_probability[:, 0])
    y_true = np.array([labels[index] for index in test_index])
    y_pred = indices_to_labels(test_pred)
    metrics = {
        "experiment": experiment["name"],
        "model": experiment["model"]["name"],
        "source_run": str(_source_dir(experiment)),
        "device": device.type,
        "n_parameters": count_parameters(build_model(experiment["model"])),
        "score_test": "fold_ensemble",
        "challenge_f1_test": challenge_f1(y_true, y_pred),
        "per_class_f1_test": per_class_f1(y_true, y_pred),
        "cv_challenge_f1_mean": float(np.mean(fold_scores)),
        "cv_challenge_f1_std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "base_fold_scores": base_fold_scores,
        "best_epochs": best_epochs,
        "confusion_matrix_test": confusion_matrix(y_true, y_pred),
        "classes": ["N", "A", "O", "~"],
        "oof_normal_to_other": int(np.sum((base_oof[splits["trainval"]] == CLASS_TO_INDEX["N"]) & (oof_pred[splits["trainval"]] == CLASS_TO_INDEX["O"]))),
        "test_normal_to_other": int(np.sum((base_test == CLASS_TO_INDEX["N"]) & (test_pred == CLASS_TO_INDEX["O"]))),
        "wall_seconds": time.perf_counter() - started,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(fold_states, run_dir / "folds.pt")
    np.savez(
        run_dir / "oof_predictions.npz",
        pred=oof_pred,
        base_pred=base_oof,
        true=np.array(labels),
        filled=oof_filled,
    )
    np.savez(
        run_dir / "test_predictions.npz",
        index=test_index,
        true=y_true,
        pred=y_pred,
        base_pred=indices_to_labels(base_test),
    )
    print(
        f"test challenge F1 {metrics['challenge_f1_test']:.4f} "
        f"cv {metrics['cv_challenge_f1_mean']:.4f} "
        f"flips test {metrics['test_normal_to_other']}",
        flush=True,
    )
    _write_result(experiment, metrics)
    return metrics


def _write_result(experiment: dict, metrics: dict) -> None:
    baseline_path = experiment["project_root"] / "results" / "11_strong_augment.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    cv_rose = metrics["cv_challenge_f1_mean"] > float(baseline["cv_challenge_f1_mean"])
    test_held = metrics["challenge_f1_test"] >= float(baseline["challenge_f1_test"])
    if cv_rose and test_held:
        decision = "Cross-validation rose and the locked test held, so the Normal review is kept."
    else:
        decision = "The Normal review is not kept. Model 11 remains the proposed model."
    payload = {
        "experiment": metrics["experiment"],
        "compared_to": "11_strong_augment",
        "change": (
            "Model 11 predictions stay in place. A record changes from Normal to Other only when "
            "a second network, trained on the usual beat and the least similar beat, says Other. "
            "The second network is fit on Normal and Other records inside each training fold."
        ),
        "model": {
            "name": metrics["model"],
            "n_parameters": metrics["n_parameters"],
        },
        "baseline_11": {
            "challenge_f1_test": baseline["challenge_f1_test"],
            "cv_challenge_f1_mean": baseline["cv_challenge_f1_mean"],
            "per_class_f1_test": baseline["per_class_f1_test"],
        },
        "challenge_f1_test": metrics["challenge_f1_test"],
        "per_class_f1_test": metrics["per_class_f1_test"],
        "cv_challenge_f1_mean": metrics["cv_challenge_f1_mean"],
        "cv_challenge_f1_std": metrics["cv_challenge_f1_std"],
        "fold_scores": metrics["fold_scores"],
        "base_fold_scores": metrics["base_fold_scores"],
        "best_epochs": metrics["best_epochs"],
        "confusion_matrix_test": metrics["confusion_matrix_test"],
        "oof_normal_to_other": metrics["oof_normal_to_other"],
        "test_normal_to_other": metrics["test_normal_to_other"],
        "wall_seconds": metrics["wall_seconds"],
        "note": decision,
    }
    path = experiment["project_root"] / "results" / "16_normal_review.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fit_fold(
    experiment: dict,
    templates: np.ndarray,
    labels: list[str],
    base_pred: np.ndarray,
    fold: dict,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, int, float, float]:
    seed_everything(seed)
    train_cfg = experiment["train"]
    train_index = np.asarray(
        [index for index in fold["train"] if labels[index] in ("N", "O")],
        dtype=int,
    )
    binary = np.asarray([0 if labels[index] == "N" else 1 for index in train_index], dtype=np.int64)
    counts = np.bincount(binary, minlength=2).astype(np.float32)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    dataset = TensorDataset(
        torch.from_numpy(templates[train_index]),
        torch.from_numpy(binary),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=len(dataset) > int(train_cfg["batch_size"]),
    )
    model = build_model(experiment["model"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(train_cfg.get("min_lr", 1e-6)),
    )
    val_index = np.asarray(fold["val"], dtype=int)
    val_templates = templates[val_index]
    base_val = base_pred[val_index]
    y_true = np.array([labels[index] for index in val_index])
    base_score = challenge_f1(y_true, indices_to_labels(base_val))
    best_score = -1.0
    best_epoch = 1
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale = 0
    patience = int(train_cfg["patience"])
    batch_size = int(train_cfg["batch_size"])
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch_templates, batch_labels in loader:
            batch_templates = batch_templates.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_templates), batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(batch_labels)
            seen += len(batch_labels)
        scheduler.step()
        probabilities = _probabilities(model, val_templates, device, batch_size)
        revised = review_normal_calls(base_val, probabilities[:, 1] >= probabilities[:, 0])
        score = challenge_f1(y_true, indices_to_labels(revised))
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch} loss {total / max(seen, 1):.4f} val F1 {score:.4f} lr {learning_rate:.6f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, float(best_score), float(base_score)


def _templates(experiment: dict, records: list[Record]) -> np.ndarray:
    representation = experiment["representation"]
    sampling_rate = int(representation["sampling_rate"])
    beat_samples = int(round(float(representation["beat_seconds"]) * sampling_rate))
    pre_samples = int(round(float(representation["pre_seconds"]) * sampling_rate))
    if beat_samples != int(experiment["model"]["beat_length"]):
        raise ValueError("Beat window and model length do not match")
    print("building two beat templates", flush=True)
    signals, masks, _inverted = ensure_signal_cache(
        records,
        experiment["preprocess_cfg"],
        experiment["project_root"] / "data" / "processed",
    )
    return two_template_batch(signals, masks, sampling_rate, beat_samples, pre_samples)


def _probabilities(
    model: nn.Module,
    templates: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(templates), batch_size):
            batch = torch.from_numpy(templates[start : start + batch_size]).to(device)
            chunks.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    if not chunks:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _other_votes(model: nn.Module, templates: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    probabilities = _probabilities(model, templates, device, batch_size)
    return probabilities[:, 1] >= probabilities[:, 0]


def _load_source(experiment: dict) -> dict:
    directory = _source_dir(experiment)
    oof = np.load(directory / "oof_predictions.npz", allow_pickle=True)
    test = np.load(directory / "test_predictions.npz", allow_pickle=True)
    return {
        "oof_pred": oof["pred"],
        "oof_filled": oof["filled"].astype(bool),
        "test_pred": test["pred"],
        "test_index": test["index"].astype(int),
    }


def _source_dir(experiment: dict) -> Path:
    source = Path(experiment["source_run"])
    if not source.is_absolute():
        source = experiment["project_root"] / source
    if not (source / "oof_predictions.npz").exists() or not (source / "test_predictions.npz").exists():
        raise FileNotFoundError(f"Missing model 11 predictions in {source}")
    return source


def _prediction_indices(pred: np.ndarray, expected: int) -> np.ndarray:
    values = np.asarray(pred)
    if values.dtype.kind in "USO":
        indices = np.asarray([CLASS_TO_INDEX[str(label)] for label in values], dtype=int)
    else:
        indices = values.astype(int)
    if len(indices) != expected:
        raise ValueError(f"Expected {expected} predictions, found {len(indices)}")
    return indices


def _test_order(saved_index: np.ndarray, split_index: np.ndarray) -> np.ndarray:
    position = {int(record): row for row, record in enumerate(saved_index)}
    missing = [int(record) for record in split_index if int(record) not in position]
    if missing:
        raise ValueError("Locked test predictions do not cover the saved split")
    return np.asarray([position[int(record)] for record in split_index], dtype=int)
