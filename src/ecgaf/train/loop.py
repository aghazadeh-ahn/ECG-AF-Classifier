"""Shared training loop for the convolutional models."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import ecgaf.models  # noqa: F401  (registers architectures)
from ecgaf.data.cache import ensure_signal_cache
from ecgaf.data.dataset import ECGDataset
from ecgaf.data.records import Record
from ecgaf.eval.metrics import (
    challenge_f1,
    confusion_matrix,
    indices_to_labels,
    per_class_f1,
)
from ecgaf.models.registry import build_model, count_parameters
from ecgaf.train.forest import load_rr_matrix
from ecgaf.train.weights import class_weights


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(dataset: ECGDataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    drop_last = shuffle and len(dataset) > batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=device.type == "cuda",
    )


def _move_batch(batch: tuple, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if len(batch) == 4:
        signal, mask, extras, label = batch
        extras = extras.to(device, non_blocking=True)
    else:
        signal, mask, label = batch
        extras = None
    signal = signal.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    return signal, mask, extras, label


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    loader.dataset.set_epoch(epoch)
    total = 0.0
    seen = 0
    for batch in loader:
        signal, mask, extras, label = _move_batch(batch, device)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signal, mask) if extras is None else model(signal, mask, extras)
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item()) * signal.shape[0]
        seen += signal.shape[0]
    return total / max(seen, 1)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            signal, mask, extras, label = _move_batch(batch, device)
            logits = model(signal, mask) if extras is None else model(signal, mask, extras)
            probability = torch.softmax(logits, dim=-1)
            probabilities.append(probability.cpu().numpy())
            predicted.append(probability.argmax(dim=-1).cpu().numpy())
            truth.append(label.numpy())
    return (
        np.concatenate(probabilities),
        np.concatenate(predicted),
        np.concatenate(truth),
    )


def _fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    train_cfg: dict,
    weights: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, int]:
    seed_everything(seed)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["patience"])
    best_score = -1.0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 1
    stale = 0
    for epoch in range(1, epochs + 1):
        loss = _train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        if val_loader is None:
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"epoch {epoch} loss {loss:.4f}", flush=True)
            continue
        probabilities, predicted, truth = _predict(model, val_loader, device)
        score = challenge_f1(indices_to_labels(truth), indices_to_labels(predicted))
        print(f"epoch {epoch} loss {loss:.4f} val F1 {score:.4f}", flush=True)
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
    return model, best_epoch


def _dataset(
    indices: list[int],
    labels: list[str],
    signals: np.ndarray,
    masks: np.ndarray,
    preprocess_cfg: dict,
    augment: bool,
    seed: int,
    rr: np.ndarray | None = None,
    rr_reference: np.ndarray | None = None,
) -> ECGDataset:
    dataset = ECGDataset(
        indices=indices,
        labels=labels,
        signals=signals,
        masks=masks,
        preprocess_cfg=preprocess_cfg,
        augment=augment,
        seed=seed,
    )
    if rr is not None and rr_reference is not None:
        reference = rr[np.asarray(rr_reference, dtype=int)]
        mean = reference.mean(axis=0)
        std = reference.std(axis=0)
        std[std < 1e-6] = 1.0
        dataset.set_rr(rr, mean, std)
    return dataset


def run_torch(experiment: dict, records: list[Record], splits: dict, run_dir: Path) -> dict:
    labels = [record.label for record in records]
    processed_dir = experiment["project_root"] / "data" / "processed"
    signals, masks, _inverted = ensure_signal_cache(records, experiment["preprocess_cfg"], processed_dir)
    rr = None
    if experiment["model"].get("use_rr"):
        rr = load_rr_matrix(records, experiment["preprocess_cfg"], experiment["project_root"])
    device = _device()
    train_cfg = experiment["train"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    seed = int(experiment["split"]["seed"])
    started = time.perf_counter()
    fold_scores: list[float] = []
    best_epochs: list[int] = []
    oof_probs = np.zeros((len(records), 4), dtype=np.float32)
    oof_pred = np.zeros(len(records), dtype=np.int64)
    oof_filled = np.zeros(len(records), dtype=bool)

    for fold_id, fold in enumerate(splits["folds"]):
        print(f"fold {fold_id}", flush=True)
        model = build_model(experiment["model"])
        train_set = _dataset(
            fold["train"], labels, signals, masks, experiment["preprocess_cfg"], True, seed + fold_id, rr, fold["train"]
        )
        val_set = _dataset(
            fold["val"], labels, signals, masks, experiment["preprocess_cfg"], False, seed, rr, fold["train"]
        )
        train_loader = _loader(train_set, batch_size, True, num_workers, device)
        val_loader = _loader(val_set, batch_size, False, num_workers, device)
        weights = class_weights(labels, fold["train"])
        model, best_epoch = _fit(model, train_loader, val_loader, train_cfg, weights, device, seed + fold_id)
        probabilities, predicted, _truth = _predict(model, val_loader, device)
        score = challenge_f1(
            np.array([labels[index] for index in fold["val"]]),
            indices_to_labels(predicted),
        )
        fold_scores.append(score)
        best_epochs.append(best_epoch)
        oof_probs[fold["val"]] = probabilities
        oof_pred[fold["val"]] = predicted
        oof_filled[fold["val"]] = True
        print(f"fold {fold_id} best epoch {best_epoch} challenge F1 {score:.4f}", flush=True)

    refit_epochs = max(1, int(np.median(best_epochs)))
    print(f"refit on trainval for {refit_epochs} epochs", flush=True)
    refit_cfg = dict(train_cfg)
    refit_cfg["epochs"] = refit_epochs
    refit_cfg["patience"] = refit_epochs + 1
    final_model = build_model(experiment["model"])
    train_set = _dataset(
        splits["trainval"],
        labels,
        signals,
        masks,
        experiment["preprocess_cfg"],
        True,
        seed + 100,
        rr,
        splits["trainval"],
    )
    train_loader = _loader(train_set, batch_size, True, num_workers, device)
    weights = class_weights(labels, splits["trainval"])
    final_model, _epoch = _fit(
        final_model, train_loader, None, refit_cfg, weights, device, seed + 100
    )
    test_set = _dataset(
        splits["test"], labels, signals, masks, experiment["preprocess_cfg"], False, seed, rr, splits["trainval"]
    )
    test_loader = _loader(test_set, batch_size, False, num_workers, device)
    timed = time.perf_counter()
    test_probs, test_pred, _test_truth = _predict(final_model, test_loader, device)
    seconds_per_record = (time.perf_counter() - timed) / max(len(splits["test"]), 1)
    y_true = np.array([labels[index] for index in splits["test"]])
    y_pred = indices_to_labels(test_pred)
    metrics = {
        "experiment": experiment["name"],
        "model": experiment["model"]["name"],
        "device": device.type,
        "n_parameters": count_parameters(final_model),
        "challenge_f1_test": challenge_f1(y_true, y_pred),
        "per_class_f1_test": per_class_f1(y_true, y_pred),
        "cv_challenge_f1_mean": float(np.mean(fold_scores)),
        "cv_challenge_f1_std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "best_epochs": best_epochs,
        "refit_epochs": refit_epochs,
        "confusion_matrix_test": confusion_matrix(y_true, y_pred),
        "classes": ["N", "A", "O", "~"],
        "seconds_per_record": seconds_per_record,
        "wall_seconds": time.perf_counter() - started,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(final_model.state_dict(), run_dir / "final.pt")
    np.savez(
        run_dir / "oof_predictions.npz",
        probs=oof_probs,
        pred=oof_pred,
        true=np.array(labels),
        filled=oof_filled,
    )
    np.savez(
        run_dir / "test_predictions.npz",
        index=np.asarray(splits["test"], dtype=np.int64),
        true=y_true,
        pred=y_pred,
        probs=test_probs.astype(np.float32),
    )
    print(
        f"test challenge F1 {metrics['challenge_f1_test']:.4f} "
        f"cv {metrics['cv_challenge_f1_mean']:.4f} params {metrics['n_parameters']}",
        flush=True,
    )
    return metrics
