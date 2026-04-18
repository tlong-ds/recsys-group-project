"""Training orchestration and local artifact registration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.evaluation import Evaluator
from recsys.models.srgnn import SRGNNRecommender
from recsys.training.registry import ModelRegistry


@dataclass
class TrainingResult:
    """Outputs produced by the training stage."""

    model: SRGNNRecommender
    artifact_path: Path
    metrics: dict[str, float]

class Trainer:
    """Fit a model, evaluate on validation set, and register artifacts locally."""
 
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        training_cfg = self.config.get("training", {})
        registry_cfg = self.config.get("registry", {})
        dvc_mode = bool(training_cfg.get("dvc_mode", False))
        registry_root = (
            registry_cfg.get("root_path")
            or training_cfg.get("registry_path")
            or "models/trained"
        )
        create_versioned = bool(registry_cfg.get("create_versioned", True))
        if dvc_mode:
            create_versioned = False
        self.registry = ModelRegistry(
            root_path=registry_root,
            create_versioned=create_versioned,
        )
 
    def train(
        self,
        model: SRGNNRecommender,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        item_vocab: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Fit the model on *train_df*, evaluate on *val_df*, register artifact.
 
        Training hyper-parameters are forwarded from ``config["training"]``
        so that ``training_config.yaml`` is the single source of truth.
        """
        if train_df.empty:
            raise ValueError("Training examples are empty.")
 
        training_cfg = self.config.get("training", {})
        fit_kwargs = {
            "num_epochs": int(training_cfg.get("num_epochs", 10)),
            "batch_size": int(training_cfg.get("batch_size", 256)),
            "lr": float(training_cfg.get("lr", 1e-3)),
            "weight_decay": float(training_cfg.get("weight_decay", 1e-5)),
            "early_stopping_patience": int(training_cfg.get("early_stopping_patience", 5)),
            "val_df": val_df if not val_df.empty else None,
            "item_vocab": item_vocab,
            "num_workers": int(training_cfg.get("num_workers", 0)),
        }
 
        fitted_model = model.fit(train_df, **fit_kwargs)
 
        # Compute offline validation metrics after training.
        metrics: dict[str, float] = {}
        if not val_df.empty:
            evaluator = Evaluator(top_k=int(training_cfg.get("top_k", 20)))
            metrics = evaluator.evaluate(fitted_model, val_df)
 
        artifact_path = self.registry.register(
            model=fitted_model,
            config=self.config,
            metrics=metrics,
        )
        return TrainingResult(model=fitted_model, artifact_path=artifact_path, metrics=metrics)
