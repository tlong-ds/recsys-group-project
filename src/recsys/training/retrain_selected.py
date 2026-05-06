from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.training.pipeline import run_training_pipeline
from recsys.utils.config import (
    load_config,
    merge_configs,
    params_to_config_overlay,
)

BEST_MODEL_PATH = Path("metrics/best_model.json")
DATA_CONFIG_PATH = Path("configs/data_config.yaml")
TRAINING_CONFIG_PATH = Path("configs/training_config.yaml")
DATA_VERSION_CONFIG_ROOT = Path("configs/data_versions")
MODEL_PROFILE_CONFIG_ROOT = Path("configs/model_profiles")
RETRAIN_DATA_ROOT = Path("data/retrained_selected")
RETRAIN_REGISTRY_ROOT = Path("models/retrained_selected")
RETRAIN_TRAIN_METRICS_PATH = Path("metrics/retrained_selected/training_metrics.json")
RETRAIN_EVAL_METRICS_PATH = Path("metrics/retrained_selected/evaluation_metrics.json")
RETRAIN_RESULT_PATH = Path("metrics/retrained_selected/retrain_result.json")


def _read_best_model_selection(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best_model = payload.get("best_model")
    if not isinstance(best_model, dict):
        raise ValueError(f"{path} must contain a 'best_model' object.")

    data_version = str(best_model.get("data_version", "")).strip()
    model_profile = str(best_model.get("model_profile", "")).strip()
    if not data_version or not model_profile:
        raise ValueError(
            "best_model must include non-empty data_version/model_profile."
        )
    return data_version, model_profile


def _load_runtime_config(
    *,
    data_config_path: Path,
    training_config_path: Path,
    model_profile_config_path: Path,
    data_version_config_path: Path,
) -> dict[str, Any]:
    base = merge_configs(
        load_config(data_config_path),
        load_config(model_profile_config_path),
        load_config(training_config_path),
    )
    data_overlay = params_to_config_overlay(load_config(data_version_config_path)).get(
        "data", {}
    )
    if not isinstance(data_overlay, dict) or not data_overlay:
        raise ValueError(f"Missing usable data overlay in {data_version_config_path}")
    return merge_configs(base, {"data": data_overlay})


def _build_train_plus_val_examples(
    *,
    train_examples_path: Path,
    val_examples_path: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    if not train_examples_path.exists():
        raise FileNotFoundError(f"Missing train examples: {train_examples_path}")
    if not val_examples_path.exists():
        raise FileNotFoundError(f"Missing val examples: {val_examples_path}")

    train_df = pd.read_parquet(train_examples_path)
    val_df = pd.read_parquet(val_examples_path)
    train_plus_val = pd.concat([train_df, val_df], ignore_index=True)

    output_root.mkdir(parents=True, exist_ok=True)
    train_plus_val_path = output_root / "trainval_examples.parquet"
    empty_val_path = output_root / "val_examples_empty.parquet"

    train_plus_val.to_parquet(train_plus_val_path, index=False)
    train_plus_val.iloc[0:0].to_parquet(empty_val_path, index=False)
    return train_plus_val_path, empty_val_path


def _augment_metrics_payload(path: Path, context: dict[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(context)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_retrain_selected_model(
    *,
    best_model_path: Path = BEST_MODEL_PATH,
    data_config_path: Path = DATA_CONFIG_PATH,
    training_config_path: Path = TRAINING_CONFIG_PATH,
    data_version_config_root: Path = DATA_VERSION_CONFIG_ROOT,
    model_profile_config_root: Path = MODEL_PROFILE_CONFIG_ROOT,
    retrain_data_root: Path = RETRAIN_DATA_ROOT,
    registry_root: Path = RETRAIN_REGISTRY_ROOT,
    train_metrics_path: Path = RETRAIN_TRAIN_METRICS_PATH,
    evaluation_metrics_path: Path = RETRAIN_EVAL_METRICS_PATH,
    output_path: Path = RETRAIN_RESULT_PATH,
) -> dict[str, Any]:
    data_version, model_profile = _read_best_model_selection(best_model_path)

    data_version_config_path = data_version_config_root / f"{data_version}.yaml"
    model_profile_config_path = model_profile_config_root / f"{model_profile}.yaml"
    if not data_version_config_path.exists():
        raise FileNotFoundError(
            f"Missing data-version config: {data_version_config_path}"
        )
    if not model_profile_config_path.exists():
        raise FileNotFoundError(
            f"Missing model-profile config: {model_profile_config_path}"
        )

    config = _load_runtime_config(
        data_config_path=data_config_path,
        training_config_path=training_config_path,
        model_profile_config_path=model_profile_config_path,
        data_version_config_path=data_version_config_path,
    )

    data_cfg = config.setdefault("data", {})
    train_examples_path = Path(str(data_cfg.get("train_examples_path", "")))
    val_examples_path = Path(str(data_cfg.get("val_examples_path", "")))
    train_plus_val_path, empty_val_path = _build_train_plus_val_examples(
        train_examples_path=train_examples_path,
        val_examples_path=val_examples_path,
        output_root=retrain_data_root,
    )

    data_cfg["train_examples_path"] = str(train_plus_val_path)
    data_cfg["val_examples_path"] = str(empty_val_path)

    training_cfg = config.setdefault("training", {})
    training_cfg["dvc_mode"] = True
    training_cfg["train_metrics_path"] = str(train_metrics_path)
    training_cfg["evaluation_metrics_path"] = str(evaluation_metrics_path)

    registry_cfg = config.setdefault("registry", {})
    registry_cfg["root_path"] = str(registry_root)

    lineage = config.setdefault("lineage", {})
    lineage["data_version"] = data_version
    lineage["data_params_path"] = str(data_version_config_path)
    lineage["selected_model_profile"] = model_profile
    lineage["selection_metrics_path"] = str(best_model_path)
    lineage["retrain_mode"] = "train_plus_val"

    result = run_training_pipeline(config)

    context = {
        "selected_model_profile": model_profile,
        "selected_data_version": data_version,
        "retrain_mode": "train_plus_val",
        "train_plus_val_examples_path": str(train_plus_val_path),
        "empty_val_examples_path": str(empty_val_path),
    }
    _augment_metrics_payload(train_metrics_path, context)
    _augment_metrics_payload(evaluation_metrics_path, context)

    payload = {
        "best_model_path": str(best_model_path),
        "data_version": data_version,
        "model_profile": model_profile,
        "model_profile_config_path": str(model_profile_config_path),
        "data_version_config_path": str(data_version_config_path),
        "train_plus_val_examples_path": str(train_plus_val_path),
        "empty_val_examples_path": str(empty_val_path),
        "training_metrics_path": str(train_metrics_path),
        "evaluation_metrics_path": str(evaluation_metrics_path),
        "model_artifact_path": result.get("artifact_path"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the selected winner profile on train+val and evaluate on test."
        )
    )
    parser.add_argument("--best-model-path", default=str(BEST_MODEL_PATH))
    parser.add_argument("--data-config", default=str(DATA_CONFIG_PATH))
    parser.add_argument("--training-config", default=str(TRAINING_CONFIG_PATH))
    parser.add_argument(
        "--data-version-config-root", default=str(DATA_VERSION_CONFIG_ROOT)
    )
    parser.add_argument(
        "--model-profile-config-root", default=str(MODEL_PROFILE_CONFIG_ROOT)
    )
    parser.add_argument("--retrain-data-root", default=str(RETRAIN_DATA_ROOT))
    parser.add_argument("--registry-root", default=str(RETRAIN_REGISTRY_ROOT))
    parser.add_argument("--train-metrics-path", default=str(RETRAIN_TRAIN_METRICS_PATH))
    parser.add_argument(
        "--evaluation-metrics-path", default=str(RETRAIN_EVAL_METRICS_PATH)
    )
    parser.add_argument("--output-path", default=str(RETRAIN_RESULT_PATH))
    args = parser.parse_args()

    result = run_retrain_selected_model(
        best_model_path=Path(args.best_model_path),
        data_config_path=Path(args.data_config),
        training_config_path=Path(args.training_config),
        data_version_config_root=Path(args.data_version_config_root),
        model_profile_config_root=Path(args.model_profile_config_root),
        retrain_data_root=Path(args.retrain_data_root),
        registry_root=Path(args.registry_root),
        train_metrics_path=Path(args.train_metrics_path),
        evaluation_metrics_path=Path(args.evaluation_metrics_path),
        output_path=Path(args.output_path),
    )
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
