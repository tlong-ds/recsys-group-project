"""End-to-end training pipeline for session-based recommendation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from recsys.data import DataLoader, DataValidator, DatasetBuilder, SessionPreprocessor
from recsys.evaluation import Evaluator
from recsys.features import SessionFeatureBuilder
from recsys.models import SRGNNRecommender
from recsys.training.trainer import Trainer
from recsys.utils.config import load_config, merge_configs
from recsys.utils.logger import get_logger


def run_training_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the training pipeline and return artifact path plus metrics."""
    logger = get_logger()
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})

    column_config = data_config.get("columns", {})
    session_col = column_config.get("session_id", "session_id")
    item_col = column_config.get("item_id", "item_id")
    timestamp_col = column_config.get("timestamp", "timestamp")

    loader = DataLoader(
        raw_path=data_config.get("raw_path", "data/raw"),
        interactions_file=data_config.get("interactions_file", "interactions.csv"),
        items_file=data_config.get("items_file", "items.csv"),
        users_file=data_config.get("users_file", "users.csv"),
    )
    validator = DataValidator(
        session_col=session_col,
        item_col=item_col,
        timestamp_col=timestamp_col,
    )
    preprocessor = SessionPreprocessor(
        session_col=session_col,
        item_col=item_col,
        timestamp_col=timestamp_col,
    )
    dataset_builder = DatasetBuilder(timestamp_col=timestamp_col)
    feature_builder = SessionFeatureBuilder(
        session_col=session_col,
        item_col=item_col,
        max_session_length=int(model_config.get("max_session_length", 20)),
    )

    interactions = loader.load_interactions()
    validator.validate_interactions(interactions)
    cleaned = preprocessor.transform(interactions)
    _persist_intermediate(cleaned, data_config)

    split = dataset_builder.build_splits(
        cleaned,
        val_ratio=float(training_config.get("val_ratio", 0.1)),
        test_ratio=float(training_config.get("test_ratio", 0.1)),
    )
    train_examples = feature_builder.build_examples(split.train)
    val_examples = feature_builder.build_examples(split.validation)
    test_examples = feature_builder.build_examples(split.test)
    _persist_processed(train_examples, val_examples, test_examples, data_config)

    model = SRGNNRecommender(
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        hidden_size=int(model_config.get("hidden_size", 128)),
        max_session_length=int(model_config.get("max_session_length", 20)),
        fallback_weight=float(model_config.get("fallback_weight", 0.15)),
        model_name=str(model_config.get("name", "srgnn")),
        model_version=str(model_config.get("version", "0.1.0")),
    )
    trainer = Trainer(config=config)
    training_result = trainer.train(model, train_examples, val_examples)

    evaluator = Evaluator(top_k=int(training_config.get("top_k", 20)))
    test_metrics = evaluator.evaluate(training_result.model, test_examples)
    logger.info("Offline test metrics: {}", test_metrics)

    return {
        "artifact_path": str(training_result.artifact_path),
        "validation_metrics": training_result.metrics,
        "test_metrics": test_metrics,
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
    result = run_training_pipeline(config)
    print(result)


def _persist_intermediate(interactions, data_config: dict[str, Any]) -> None:
    interim_path = Path(data_config.get("interim_path", "data/interim"))
    interim_path.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(interim_path / "clean_interactions.csv", index=False)


def _persist_processed(train_examples, val_examples, test_examples, data_config: dict[str, Any]) -> None:
    processed_path = Path(data_config.get("processed_path", "data/processed"))
    processed_path.mkdir(parents=True, exist_ok=True)
    train_examples.to_json(processed_path / "train_examples.jsonl", orient="records", lines=True)
    val_examples.to_json(processed_path / "val_examples.jsonl", orient="records", lines=True)
    test_examples.to_json(processed_path / "test_examples.jsonl", orient="records", lines=True)
