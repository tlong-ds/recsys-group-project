"""Config loader: reads YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return its contents as a dict."""
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return loaded


def load_optional_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file if it exists, otherwise return an empty mapping."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    loaded = load_config(cfg_path)
    return loaded if isinstance(loaded, dict) else {}


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge multiple config dicts (later configs take precedence)."""
    result: dict[str, Any] = {}
    for cfg in configs:
        _deep_merge(result, cfg)
    return result


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


_DATA_PARAM_KEYS = (
    "raw_path",
    "interim_path",
    "processed_path",
    "train_examples_path",
    "val_examples_path",
    "test_examples_path",
    "item_vocab_path",
    "validation",
    "temporal_split",
    "augmentation",
    "training_example_format",
    "parquet_compression",
    "logging",
    "compatibility",
)
_TRAINING_PARAM_KEYS = (
    "seed",
    "test_ratio",
    "val_ratio",
    "batch_size",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
    "num_epochs",
    "lr",
    "weight_decay",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "top_k",
    "device",
)


def params_to_config_overlay(params_cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert params.yaml into a strict experiment-only overlay mapping."""
    overlay: dict[str, Any] = {}

    data_cfg = params_cfg.get("data")
    if isinstance(data_cfg, dict):
        data_overlay = {k: data_cfg[k] for k in _DATA_PARAM_KEYS if k in data_cfg}
        if data_overlay:
            overlay["data"] = data_overlay

    model_cfg = params_cfg.get("model")
    if isinstance(model_cfg, dict):
        overlay["model"] = dict(model_cfg)

    training_cfg = params_cfg.get("training")
    if isinstance(training_cfg, dict):
        training_overlay = {
            k: training_cfg[k] for k in _TRAINING_PARAM_KEYS if k in training_cfg
        }
        if training_overlay:
            overlay["training"] = training_overlay

    return overlay


def load_data_config_with_params(
    data_config_path: str | Path,
    params_path: str | Path = "params.yaml",
) -> dict[str, Any]:
    """Load data config and apply params.yaml overlay for the data section."""
    data_base = load_config(data_config_path).get("data")
    if not isinstance(data_base, dict):
        raise ValueError(f"Missing 'data' section in config: {data_config_path}")

    params_cfg = load_optional_config(params_path)
    params_overlay = params_to_config_overlay(params_cfg).get("data", {})
    if not isinstance(params_overlay, dict):
        params_overlay = {}
    return merge_configs(data_base, params_overlay)


def load_training_runtime_config(
    *,
    data_config_path: str | Path,
    model_config_path: str | Path,
    training_config_path: str | Path,
    params_path: str | Path = "params.yaml",
    data_params_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load merged runtime config from base config files plus params overlay."""
    base = merge_configs(
        load_config(data_config_path),
        load_config(model_config_path),
        load_config(training_config_path),
    )
    params_overlay = params_to_config_overlay(load_optional_config(params_path))
    merged = merge_configs(base, params_overlay)

    if data_params_path is not None:
        data_overlay = params_to_config_overlay(load_config(data_params_path)).get(
            "data", {}
        )
        if isinstance(data_overlay, dict) and data_overlay:
            merged = merge_configs(merged, {"data": data_overlay})

    return merged
