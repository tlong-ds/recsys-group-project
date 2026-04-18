"""Training orchestration and local artifact registration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.evaluation import Evaluator
from recsys.models.graph_helpers import GraphRecommenderBase
from recsys.training.registry import ModelRegistry


@dataclass
class TrainingResult:
    """Outputs produced by the training stage."""

    model:         GraphRecommenderBase
    artifact_path: Path
    metrics:       dict[str, float]


class Trainer:
    """Fit a model, evaluate on validation set, and register artifacts locally.

    Accepts any ``GraphRecommenderBase`` subclass
    (SRGNNRecommender, TAGNNRecommender, GGNNRecommender) — the trainer
    delegates entirely to the model's own ``fit`` and ``save`` methods,
    so no architecture-specific logic lives here.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config      = config
        training_cfg     = config.get("training", {})
        registry_cfg     = config.get("registry", {})
        dvc_mode         = bool(training_cfg.get("dvc_mode", False))
        registry_root    = (
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
        model:      GraphRecommenderBase,
        train_df:   pd.DataFrame,
        val_df:     pd.DataFrame,
        item_vocab: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Fit *model* on *train_df*, evaluate on *val_df*, register artifact.

        Training hyper-parameters are forwarded from ``config["training"]``
        so that ``training_config.yaml`` is the single source of truth.

        Parameters
        ----------
        model:
            An *unfitted* ``GraphRecommenderBase`` instance (any architecture).
        train_df:
            Processed training examples (parquet schema).
        val_df:
            Processed validation examples.  Pass an empty DataFrame to skip
            validation and early stopping.
        item_vocab:
            Optional ``{"item2id": {...}}`` mapping loaded from ``item_vocab.json``.
        """
        if train_df.empty:
            raise ValueError("Training examples are empty.")

        training_cfg = self.config.get("training", {})
        fit_kwargs: dict[str, Any] = {
            "num_epochs":               int(training_cfg.get("num_epochs",              10)),
            "batch_size":               int(training_cfg.get("batch_size",             256)),
            "lr":                     float(training_cfg.get("lr",                    1e-3)),
            "weight_decay":           float(training_cfg.get("weight_decay",          1e-5)),
            "early_stopping_patience":  int(training_cfg.get("early_stopping_patience",  5)),
            "val_df":                   val_df if not val_df.empty else None,
            "item_vocab":               item_vocab,
            "num_workers":              int(training_cfg.get("num_workers",              0)),
        }

        fitted_model = model.fit(train_df, **fit_kwargs)

        # Compute offline validation metrics after training
        metrics: dict[str, float] = {}
        if not val_df.empty:
            evaluator = Evaluator(top_k=int(training_cfg.get("top_k", 20)))
            metrics   = evaluator.evaluate(fitted_model, val_df)

        artifact_path = self.registry.register(
            model=fitted_model,
            config=self.config,
            metrics=metrics,
        )
        return TrainingResult(model=fitted_model, artifact_path=artifact_path, metrics=metrics)