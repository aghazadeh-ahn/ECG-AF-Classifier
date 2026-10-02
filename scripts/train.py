"""Train or score one experiment file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from ecgaf.config import load_experiment, project_root
from ecgaf.data.records import load_records
from ecgaf.eval.metrics import challenge_f1
from ecgaf.eval.selective import choose_threshold, selective_scores
from ecgaf.eval.splits import load_or_create_splits
from ecgaf.train.forest import run_forest
from ecgaf.train.loop import run_torch


def run_selective(experiment: dict) -> dict:
    source = Path(experiment["source_run"])
    if not source.is_absolute():
        source = project_root() / source
    oof_path = source / "oof_predictions.npz"
    test_path = source / "test_predictions.npz"
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing predictions in {source}. Train the proposed model before scoring abstention."
        )
    oof = np.load(oof_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    filled = oof["filled"].astype(bool)
    oof_true = oof["true"][filled]
    oof_pred_index = oof["pred"][filled]
    from ecgaf.eval.metrics import indices_to_labels

    oof_pred = indices_to_labels(oof_pred_index)
    oof_confidence = oof["probs"][filled].max(axis=1)
    selective_cfg = experiment["selective"]
    selection = choose_threshold(
        oof_true,
        oof_pred,
        oof_confidence,
        thresholds=[float(value) for value in selective_cfg["thresholds"]],
        min_coverage=float(selective_cfg["min_coverage"]),
    )
    threshold = float(selection["chosen"]["threshold"])
    test_confidence = test["probs"].max(axis=1)
    test_scores = selective_scores(test["true"], test["pred"], test_confidence, threshold)
    # The official number stays the score on every test record.
    test_scores["challenge_f1_all"] = challenge_f1(test["true"], test["pred"])
    metrics = {
        "experiment": experiment["name"],
        "source_run": str(source),
        "threshold_selected_on": "out_of_fold_validation",
        "selection": selection,
        "test": test_scores,
    }
    run_dir = project_root() / "runs" / experiment["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"threshold {threshold:.2f} coverage {test_scores['coverage']:.3f} "
        f"F1 all {test_scores['challenge_f1_all']:.4f} "
        f"F1 accepted {test_scores['challenge_f1_accepted']:.4f}",
        flush=True,
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: python scripts/train.py configs/experiments/<name>.yaml")
    experiment = load_experiment(Path(args[0]))
    if experiment.get("eval_only"):
        run_selective(experiment)
        return
    records = load_records(experiment["data_root"])
    labels = [record.label for record in records]
    split_cfg = experiment["split"]
    splits = load_or_create_splits(
        labels,
        processed_dir=project_root() / "data" / "processed",
        seed=int(split_cfg["seed"]),
        test_size=float(split_cfg["test_size"]),
        n_folds=int(split_cfg["n_folds"]),
    )
    run_dir = project_root() / "runs" / experiment["name"]
    if experiment["model"]["name"] == "random_forest":
        run_forest(experiment, records, splits, run_dir)
    else:
        run_torch(experiment, records, splits, run_dir)


if __name__ == "__main__":
    main()
