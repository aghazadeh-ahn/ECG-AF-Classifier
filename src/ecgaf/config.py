"""Load experiment YAML and merge preprocessing overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_experiment(path: Path) -> dict[str, Any]:
    path = path.resolve()
    experiment = load_yaml(path)
    root = project_root()
    if "preprocess" in experiment:
        preprocess_path = Path(experiment["preprocess"])
        if not preprocess_path.is_absolute():
            preprocess_path = root / preprocess_path
        preprocess = load_yaml(preprocess_path)
        overrides = experiment.get("preprocess_overrides") or {}
        experiment["preprocess_cfg"] = deep_merge(preprocess, overrides)
    experiment["project_root"] = root
    if "data_root" in experiment:
        data_root = Path(experiment["data_root"])
        if not data_root.is_absolute():
            data_root = root / data_root
        experiment["data_root"] = data_root
    return experiment
