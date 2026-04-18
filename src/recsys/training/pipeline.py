"""Training pipeline for SR-GNN using preprocessed parquet graph examples."""

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

from recsys.data.pipeline import DataProcessingPipeline
from recsys.evaluation import Evaluator
from recsys.models import SRGNNRecommender
from recsys.training.trainer import Trainer
from recsys.utils.config import load_config, merge_configs
from recsys.utils.logger import get_logger


def run_training_pipeline(config: dict[str, Any], data_config_path: str | Path) -> dict[str, Any]:
    """Execute training directly from processed parquet examples."""
    logger = get_logger()
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    mlflow_config = config.get("mlflow", {})

    seed = int(training_config.get("seed", 42))
    _set_seed(seed)

    _run_data_pipeline(data_config_path, logger)

    train_df = _load_examples(
        _resolve_path(data_config.get("train_examples_path", "data/processed/train_examples.parquet"))
    )
    val_df = _load_examples(
        _resolve_path(data_config.get("val_examples_path", "data/processed/val_examples.parquet"))
    )
    test_df = _load_examples(
        _resolve_path(data_config.get("test_examples_path", "data/processed/test_examples.parquet"))
    )
    item_vocab = _load_item_vocab(
        _resolve_path(data_config.get("item_vocab_path", "data/processed/item_vocab.json"))
    )

    model = SRGNNRecommender(
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        hidden_size=int(model_config.get("hidden_size", 128)),
        step=int(model_config.get("step", 1)),
        max_session_length=int(model_config.get("max_session_length", 20)),
        fallback_weight=float(model_config.get("fallback_weight", 0.0)),
        model_name=str(model_config.get("name", "srgnn")),
        model_version=str(model_config.get("version", "0.1.0")),
        seed=seed,
    )

    trainer = Trainer(config=config)
    mlflow_context = _mlflow_run_context(config) if mlflow_config.get("enabled", False) else nullcontext()
    with mlflow_context as active_run:
        if active_run is not None:
            _log_config_to_mlflow(config)

        training_result = trainer.train(model, train_df, val_df, item_vocab=item_vocab)

        evaluator = Evaluator(top_k=int(training_config.get("top_k", 20)))
        test_metrics = evaluator.evaluate(training_result.model, test_df)
        logger.info("offline test metrics: {}", test_metrics)

        if active_run is not None:
            mlflow = _get_mlflow()
            mlflow.log_metrics({f"val_{key}": value for key, value in training_result.metrics.items()})
            mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
            mlflow.log_artifacts(str(training_result.artifact_path.parent), artifact_path="registered_model")

            if training_result.model._core is not None:
                mlflow.pytorch.log_model(training_result.model._core, artifact_path="model_core")

        return {
            "artifact_path": str(training_result.artifact_path),
            "validation_metrics": training_result.metrics,
            "test_metrics": test_metrics,
            "mlflow_run_id": active_run.info.run_id if active_run is not None else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SR-GNN recommender")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--training-config", default="configs/training_config.yaml")
    args = parser.parse_args()

    config = merge_configs(
        load_config(args.data_config),
        load_config(args.model_config),
        load_config(args.training_config),
    )
    result = run_training_pipeline(config, data_config_path=args.data_config)
    print(result)


def _run_data_pipeline(data_config_path: str | Path, logger) -> None:
    """Run the full data processing pipeline before training."""
    logger.info("Running data pipeline from config: {}", data_config_path)
    try:
        pipeline = DataProcessingPipeline(config_path=data_config_path)
        outputs = pipeline.run()
        logger.info("Data pipeline complete, produced {} artifacts.", len(outputs))
    except Exception as exc:
        logger.error("Data pipeline failed: {}", exc)
        raise


def _resolve_path(path_value: str) -> Path:
    return Path(path_value)


def _load_examples(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed examples not found: {path}")
    return pd.read_parquet(path)


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
    mlflow = _get_mlflow()
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment_name = mlflow_config.get("experiment_name")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=mlflow_config.get("run_name"))


def _log_config_to_mlflow(config: dict[str, Any]) -> None:
    mlflow = _get_mlflow()
    flattened = _flatten_dict(config)
    for key, value in flattened.items():
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


def _get_mlflow():
    import mlflow

    return mlflow


if __name__ == "__main__":
    main()