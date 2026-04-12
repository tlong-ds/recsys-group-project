"""Offline evaluator for next-item prediction."""

from __future__ import annotations

from collections.abc import Sequence
import pandas as pd

from recsys.evaluation.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k
from recsys.models.srgnn import SRGNNRecommender


class Evaluator:
    """Compute ranking metrics on held-out next-item examples."""

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    def evaluate(
        self,
        model: SRGNNRecommender,
        examples: pd.DataFrame,
    ) -> dict[str, float]:
        """Return average HR@K, MRR@K, and NDCG@K."""
        if examples.empty:
            return {"hr@k": 0.0, "mrr@k": 0.0, "ndcg@k": 0.0}

        hr_scores: list[float] = []
        mrr_scores: list[float] = []
        ndcg_scores: list[float] = []

        for row in examples.itertuples(index=False):
            context_items = self._coerce_context(row.context_items)
            target_item = int(row.target_item)
            recommendations = model.recommend(context_items, top_k=self.top_k)
            hr_scores.append(hit_rate_at_k(recommendations, [target_item], self.top_k))
            mrr_scores.append(mrr_at_k(recommendations, [target_item], self.top_k))
            ndcg_scores.append(ndcg_at_k(recommendations, [target_item], self.top_k))

        return {
            "hr@k": sum(hr_scores) / len(hr_scores),
            "mrr@k": sum(mrr_scores) / len(mrr_scores),
            "ndcg@k": sum(ndcg_scores) / len(ndcg_scores),
        }

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Return metrics unchanged for higher-level logging hooks."""
        return None

    @staticmethod
    def _coerce_context(value: Sequence[int] | str) -> list[int]:
        if isinstance(value, str):
            stripped = value.strip().strip("[]")
            if not stripped:
                return []
            return [int(part.strip()) for part in stripped.split(",")]
        return [int(item) for item in value]
