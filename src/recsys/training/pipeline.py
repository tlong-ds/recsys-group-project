"""Training pipeline – supports SR-GNN variants, TAGNN, and GGNN.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from recsys.evaluation import Evaluator
from recsys.models.graph_helpers import GraphRecommenderBase
from recsys.models.ggnn import GGNNRecommender
from recsys.models.srgnn import SRGNNRecommender, VARIANT_SRGNN
from recsys.models.tagnn import TAGNNRecommender
from recsys.training.mlflow_registry import register_model_version
from recsys.training.registry import ModelRegistry
from recsys.training.tracking import (
    configure_system_metrics,
    configure_tracking,
    log_evaluation_run,
    sanitize_metric_key,
    system_metrics_run_override,
    tracking_enabled,
)
from recsys.training.trainer import Trainer
from recsys.utils.config import load_training_runtime_config
from recsys.utils.logger import get_logger

STAGE_ALL      = "all"
STAGE_TRAIN    = "train"
STAGE_EVALUATE = "evaluate"
STAGE_NAMES    = [STAGE_ALL, STAGE_TRAIN, STAGE_EVALUATE]

# Supported model type strings
MODEL_TYPE_SRGNN = "srgnn"
MODEL_TYPE_TAGNN = "tagnn"
MODEL_TYPE_GGNN  = "ggnn"


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(model_config: dict[str, Any], seed: int) -> GraphRecommenderBase:
    """Instantiate the correct model class from ``model_config``.

    The ``type`` key selects the architecture; ``variant`` is passed through
    for SR-GNN only (ignored for TAGNN / GGNN).

    Parameters
    ----------
    model_config:
        Contents of ``config["model"]``.
    seed:
        Global random seed forwarded to the model.
    """
    model_type = str(model_config.get("type", MODEL_TYPE_SRGNN)).lower()

    common_kwargs: dict[str, Any] = dict(
        embedding_dim      = int(model_config.get("embedding_dim", 128)),
        hidden_size        = int(model_config.get("hidden_size", 128)),
        step               = int(model_config.get("step", 1)),
        max_session_length = int(model_config.get("max_session_length", 20)),
        fallback_weight    = float(model_config.get("fallback_weight", 0.0)),
        model_version      = str(model_config.get("version", "0.1.0")),
        seed               = seed,
    )

    # model_name defaults: explicit > variant (srgnn) > type
    explicit_name = model_config.get("name")

    if model_type == MODEL_TYPE_TAGNN:
        from recsys.models.tagnn import DEFAULT_SCORE_CHUNK
        common_kwargs["model_name"]       = explicit_name or "tagnn"
        common_kwargs["score_chunk_size"] = int(
            model_config.get("score_chunk_size", DEFAULT_SCORE_CHUNK)
        )
        return TAGNNRecommender(**common_kwargs)

    if model_type == MODEL_TYPE_GGNN:
        common_kwargs["model_name"] = explicit_name or "ggnn"
        return GGNNRecommender(**common_kwargs)

    if model_type not in (MODEL_TYPE_SRGNN, MODEL_TYPE_TAGNN, MODEL_TYPE_GGNN):
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose one of: {MODEL_TYPE_SRGNN!r}, {MODEL_TYPE_TAGNN!r}, {MODEL_TYPE_GGNN!r}"
        )

    # Default: SR-GNN
    variant = str(model_config.get("variant", VARIANT_SRGNN))
    common_kwargs["variant"]    = variant
    common_kwargs["model_name"] = explicit_name or variant
    return SRGNNRecommender(**common_kwargs)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def run_train_stage(config: dict[str, Any]) -> dict[str, Any]:
    """Train and register a model artefact from processed train/val examples."""
    logger          = get_logger()
    data_config     = config.get("data", {})
    model_config    = config.get("model", {})
    training_config = config.get("training", {})

    seed = int(training_config.get("seed", 42))
    _set_seed(seed)

    train_df   = _load_split_examples(data_config, "train")
    val_df     = _load_split_examples(data_config, "val")
    item_vocab = _load_item_vocab(_resolve_item_vocab_path(data_config))

    model = build_model(model_config, seed)

    trainer         = Trainer(config=config)
    mlflow_context  = _mlflow_run_context(config) if tracking_enabled(config) else nullcontext()
    registry_info: dict[str, Any] | None = None

    with mlflow_context as active_run:
        if active_run is not None:
            _log_config_to_mlflow(config)

        training_result = trainer.train(model, train_df, val_df, item_vocab=item_vocab)

        logger.info("offline validation metrics: {}", training_result.metrics)

        if active_run is not None:
            mlflow = _get_mlflow()
            mlflow.log_metrics(
                {
                    f"val_{sanitize_metric_key(k)}": float(v)
                    for k, v in training_result.metrics.items()
                }
            )
            mlflow.log_artifacts(
                str(training_result.artifact_path.parent),
                artifact_path="registered_model",
            )
            if training_result.model._core is not None:
                mlflow.pytorch.log_model(
                    training_result.model._core,
                    artifact_path="model_core",
                )
                registry_info = register_model_version(
                    config=config,
                    run_id=active_run.info.run_id,
                    source_artifact_path="model_core",
                )

        metrics_path = _metrics_path(
            training_config, "train_metrics_path", "metrics/training_metrics.json"
        )
        payload: dict[str, Any] = {
            "artifact_path":      str(training_result.artifact_path),
            "validation_metrics": training_result.metrics,
            "mlflow_run_id":      active_run.info.run_id if active_run is not None else None,
        }
        if registry_info is not None:
            payload["model_registry"] = registry_info
        _write_json(payload, metrics_path)

    outputs: dict[str, Any] = {
        "artifact_path":      str(training_result.artifact_path),
        "validation_metrics": training_result.metrics,
        "training_metrics":   str(metrics_path),
    }
    if registry_info is not None:
        outputs["model_registry"] = registry_info
    return outputs


def run_evaluate_stage(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the latest registered model on test examples."""
    logger          = get_logger()
    data_cfg        = config.get("data", {})
    training_cfg    = config.get("training", {})
    model_config    = config.get("model", {})
    registry_root   = (
        config.get("registry", {}).get("root_path")
        or training_cfg.get("registry_path")
        or "models/trained"
    )

    model_path    = ModelRegistry(root_path=registry_root).latest_model_path()
    model_type    = str(model_config.get("type", MODEL_TYPE_SRGNN)).lower()
    model         = _load_model_by_type(model_type, model_path)
    test_examples = _load_split_examples(data_cfg, "test")

    evaluator    = Evaluator(top_k=int(training_cfg.get("top_k", 20)))
    test_metrics = evaluator.evaluate(model, test_examples)
    logger.info("offline test metrics: {}", test_metrics)
    log_evaluation_run(config=config, metrics=test_metrics, model_path=model_path)

    metrics_path = _metrics_path(
        training_cfg, "evaluation_metrics_path", "metrics/evaluation_metrics.json"
    )
    _write_json({"model_path": str(model_path), "test_metrics": test_metrics}, metrics_path)

    return {
        "model_path":         str(model_path),
        "test_metrics":       test_metrics,
        "evaluation_metrics": str(metrics_path),
    }


def run_training_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full train + evaluate workflow."""
    outputs = run_train_stage(config)
    outputs.update(run_evaluate_stage(config))
    return outputs


# ---------------------------------------------------------------------------
# Model loading helper (selects correct load() classmethod by type)
# ---------------------------------------------------------------------------


def _load_model_by_type(
    model_type: str, model_path: Path
) -> GraphRecommenderBase:
    if model_type == MODEL_TYPE_TAGNN:
        return TAGNNRecommender.load(model_path)
    if model_type == MODEL_TYPE_GGNN:
        return GGNNRecommender.load(model_path)
    return SRGNNRecommender.load(model_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run graph recommender training")
    parser.add_argument("--data-config",     default="configs/data_config.yaml")
    parser.add_argument("--model-config",    default="configs/model_config.yaml")
    parser.add_argument("--training-config", default="configs/training_config.yaml")
    parser.add_argument("--params",          default="params.yaml")
    parser.add_argument("--stage",           default=STAGE_ALL, choices=STAGE_NAMES)
    parser.add_argument(
        "--dvc-mode",
        action="store_true",
        help="Disable versioned registry dirs for deterministic DVC outputs.",
    )
    args = parser.parse_args()

    config = load_training_runtime_config(
        data_config_path     = args.data_config,
        model_config_path    = args.model_config,
        training_config_path = args.training_config,
        params_path          = args.params,
    )
    if args.dvc_mode:
        config.setdefault("training", {})["dvc_mode"] = True

    if args.stage == STAGE_ALL:
        result = run_training_pipeline(config)
    elif args.stage == STAGE_TRAIN:
        result = run_train_stage(config)
    elif args.stage == STAGE_EVALUATE:
        result = run_evaluate_stage(config)
    else:
        raise ValueError(f"Unsupported stage '{args.stage}'")
    print(result)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_split_examples(data_config: dict[str, Any], split: str) -> pd.DataFrame:
    return _load_examples(_resolve_split_path(data_config, split))


def _resolve_split_path(data_config: dict[str, Any], split: str) -> Path:
    explicit = data_config.get(f"{split}_examples_path")
    if explicit:
        return Path(str(explicit))
    processed_dir = Path(str(data_config.get("processed_path", "data/processed")))
    return processed_dir / f"{split}_examples.parquet"


def _resolve_item_vocab_path(data_config: dict[str, Any]) -> Path:
    explicit = data_config.get("item_vocab_path")
    if explicit:
        return Path(str(explicit))
    processed_dir = Path(str(data_config.get("processed_path", "data/processed")))
    return processed_dir / "item_vocab.json"


def _load_examples(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed examples not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported training example format: {path.suffix}")


def _load_item_vocab(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Item vocab not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mlflow_run_context(config: dict[str, Any]):
    configure_tracking(config)
    configure_system_metrics(config)
    mlflow         = _get_mlflow()
    mlflow_cfg     = config.get("mlflow", {})
    run_kwargs: dict[str, Any] = {"run_name": mlflow_cfg.get("run_name")}
    lsm            = system_metrics_run_override(config)
    if lsm is not None:
        run_kwargs["log_system_metrics"] = lsm
    return mlflow.start_run(**run_kwargs)


def _log_config_to_mlflow(config: dict[str, Any]) -> None:
    mlflow   = _get_mlflow()
    for key, value in _flatten_dict(config).items():
        mlflow.log_param(key, value)


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, full_key))
        else:
            flat[full_key] = str(value)
    return flat


def _metrics_path(training_cfg: dict[str, Any], key: str, default: str) -> Path:
    return Path(str(training_cfg.get(key, default)))


def _write_json(payload: dict[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def _get_mlflow():
    import mlflow
    return mlflow


if __name__ == "__main__":
    main()