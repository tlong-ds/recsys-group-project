"""Offline evaluator for graph-based next-item prediction."""

from __future__ import annotations

import pandas as pd

from recsys.evaluation.metrics import hit_rate_at_k, mrr_at_k
from recsys.models.graph_helpers import GraphRecommenderBase


class Evaluator:
    """Compute ranking metrics on held-out graph examples.

    Works with any model that inherits from ``GraphRecommenderBase``
    (SRGNNRecommender, TAGNNRecommender, GGNNRecommender).

    ``recommend_from_graph`` is called with the pre-built graph tensors
    stored in each example row, so the evaluator never needs to know which
    model architecture it is measuring.
    """

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    def evaluate(
        self,
        model: GraphRecommenderBase,
        examples: pd.DataFrame,
    ) -> dict[str, float]:
        """Return HR@K and MRR@K over *examples*.

        Parameters
        ----------
        model:
            Any fitted ``GraphRecommenderBase`` subclass.
        examples:
            DataFrame with columns ``x``, ``edge_index``, ``alias_inputs``,
            ``pos_items`` (same schema as the parquet training splits).
        """
        if examples.empty:
            return {"hr@k": 0.0, "mrr@k": 0.0}

        hr_scores:  list[float] = []
        mrr_scores: list[float] = []

        for row in examples.itertuples(index=False):
            recommendations = model.recommend_from_graph(
                row.x, row.edge_index, row.alias_inputs, top_k=self.top_k
            )
            target_item = int(row.pos_items)
            hr_scores.append(hit_rate_at_k(recommendations, [target_item], self.top_k))
            mrr_scores.append(mrr_at_k(recommendations, [target_item], self.top_k))

        count = len(hr_scores)
        return {
            "hr@k":  sum(hr_scores)  / count,
            "mrr@k": sum(mrr_scores) / count,
        }

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """No-op hook; override in subclasses to emit metrics to a tracker."""
        return None