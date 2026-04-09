"""Evaluator: runs a model against a held-out test set and reports metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from recsys.models.base_model import BaseRecsysModel


class Evaluator:
    """Compute recommendation quality metrics on test data."""

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    def evaluate(
        self,
        model: BaseRecsysModel,
        test_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Return a dict of metric_name → score."""
        # TODO: implement metric computation
        raise NotImplementedError

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to MLflow or stdout."""
        # TODO: implement
        raise NotImplementedError
