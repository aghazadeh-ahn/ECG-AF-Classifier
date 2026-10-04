"""Random-forest baseline on RR features, scored with the challenge F1."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ecgaf.data.cache import preprocess_cache_key
from ecgaf.data.records import Record, load_signal
from ecgaf.eval.metrics import CLASS_TO_INDEX, challenge_f1, confusion_matrix, per_class_f1
from ecgaf.features.rr import FEATURE_NAMES, rr_features
from ecgaf.preprocess.pipeline import preprocess_signal


def _feature_matrix(records: list[Record], preprocess_cfg: dict, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(records):
            return cached
    cfg = preprocess_cfg
    matrix = np.zeros((len(records), len(FEATURE_NAMES)), dtype=np.float32)
    for index, record in enumerate(records):
        raw, sampling_rate = load_signal(record)
        processed = preprocess_signal(raw, sampling_rate, cfg, augment=False)
        matrix[index] = rr_features(processed.signal, sampling_rate)
        if index % 500 == 0:
            print(f"rr features {index}/{len(records)}", flush=True)
    matrix = np.nan_to_num(matrix)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    return matrix


def _fit_forest(features: np.ndarray, labels: np.ndarray, model_cfg: dict, seed: int) -> RandomForestClassifier:
    depth = model_cfg.get("max_depth")
    forest = RandomForestClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 300)),
        max_depth=None if depth in (None, "null") else int(depth),
        class_weight=model_cfg.get("class_weight", "balanced"),
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(features, labels)
    return forest


def load_rr_matrix(records: list[Record], preprocess_cfg: dict, project_root: Path) -> np.ndarray:
    feature_cfg = json.loads(json.dumps(preprocess_cfg))
    feature_cfg["length"] = dict(feature_cfg["length"])
    feature_cfg["length"]["enabled"] = False
    feature_cfg["augment"] = dict(feature_cfg.get("augment", {}))
    feature_cfg["augment"]["enabled"] = False
    cache_path = project_root / "data" / "processed" / "features" / f"rr_{preprocess_cache_key(feature_cfg)}.npy"
    return _feature_matrix(records, feature_cfg, cache_path)


def run_forest(experiment: dict, records: list[Record], splits: dict, run_dir: Path) -> dict:
    labels = np.array([record.label for record in records])
    feature_cfg = json.loads(json.dumps(experiment["preprocess_cfg"]))
    feature_cfg["length"] = dict(feature_cfg["length"])
    feature_cfg["length"]["enabled"] = False
    feature_cfg["augment"] = dict(feature_cfg.get("augment", {}))
    feature_cfg["augment"]["enabled"] = False
    cache_path = (
        experiment["project_root"]
        / "data"
        / "processed"
        / "features"
        / f"rr_{preprocess_cache_key(feature_cfg)}.npy"
    )
    started = time.perf_counter()
    features = _feature_matrix(records, feature_cfg, cache_path)
    model_cfg = experiment["model"]
    seed = int(experiment["split"]["seed"])
    fold_scores = []
    for fold_id, fold in enumerate(splits["folds"]):
        forest = _fit_forest(features[fold["train"]], labels[fold["train"]], model_cfg, seed + fold_id)
        predicted = forest.predict(features[fold["val"]])
        score = challenge_f1(labels[fold["val"]], predicted)
        fold_scores.append(score)
        print(f"fold {fold_id} challenge F1 {score:.4f}", flush=True)

    forest = _fit_forest(features[splits["trainval"]], labels[splits["trainval"]], model_cfg, seed)
    test_index = np.asarray(splits["test"])
    predict_started = time.perf_counter()
    predicted = forest.predict(features[test_index])
    seconds_per_record = (time.perf_counter() - predict_started) / max(len(test_index), 1)
    probabilities = forest.predict_proba(features[test_index])
    class_order = list(forest.classes_)
    full_probabilities = np.zeros((len(test_index), 4), dtype=np.float32)
    for column, label in enumerate(class_order):
        full_probabilities[:, CLASS_TO_INDEX[label]] = probabilities[:, column]
    y_true = labels[test_index]
    metrics = {
        "experiment": experiment["name"],
        "challenge_f1_test": challenge_f1(y_true, predicted),
        "per_class_f1_test": per_class_f1(y_true, predicted),
        "cv_challenge_f1_mean": float(np.mean(fold_scores)),
        "cv_challenge_f1_std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "confusion_matrix_test": confusion_matrix(y_true, predicted),
        "classes": ["N", "A", "O", "~"],
        "seconds_per_record": seconds_per_record,
        "wall_seconds": time.perf_counter() - started,
        "n_parameters": None,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez(
        run_dir / "test_predictions.npz",
        index=test_index,
        true=y_true,
        pred=predicted,
        probs=full_probabilities,
    )
    print(
        f"test challenge F1 {metrics['challenge_f1_test']:.4f} "
        f"cv {metrics['cv_challenge_f1_mean']:.4f}",
        flush=True,
    )
    return metrics
