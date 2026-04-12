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
    """Fit a model, evaluate it, and register artifacts locally."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        registry_root = (
            self.config.get("registry", {}).get("root_path")
            or self.config.get("training", {}).get("registry_path")
            or "models/trained"
        )
        self.registry = ModelRegistry(registry_root)

    def train(
        self,
        model: SRGNNRecommender,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> TrainingResult:
        """Fit the model and register the resulting artifact."""
        if train_df.empty:
            raise ValueError("Training examples are empty")

        fitted_model = model.fit(train_df)
        metrics = {}
        if not val_df.empty:
            evaluator = Evaluator(top_k=int(self.config.get("training", {}).get("top_k", 20)))
            metrics = evaluator.evaluate(fitted_model, val_df)

        artifact_path = self.registry.register(
            model=fitted_model,
            config=self.config,
            metrics=metrics,
        )
        return TrainingResult(model=fitted_model, artifact_path=artifact_path, metrics=metrics)
