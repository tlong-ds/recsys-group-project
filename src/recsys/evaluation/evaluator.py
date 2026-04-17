"""Offline evaluator for graph-based next-item prediction."""

from __future__ import annotations

import pandas as pd

from recsys.evaluation.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k
from recsys.models.srgnn import SRGNNRecommender


class Evaluator:
    """Compute ranking metrics on held-out graph examples."""

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    def evaluate(
        self,
        model: SRGNNRecommender,
        examples: pd.DataFrame,
    ) -> dict[str, float]:
        if examples.empty:
            return {"hr@k": 0.0, "mrr@k": 0.0, "ndcg@k": 0.0}

        hr_scores: list[float] = []
        mrr_scores: list[float] = []
        ndcg_scores: list[float] = []

        for row in examples.itertuples(index=False):
            recommendations = model.recommend_from_graph(row.x, row.alias_inputs, top_k=self.top_k)
            target_item = int(row.pos_items)
            hr_scores.append(hit_rate_at_k(recommendations, [target_item], self.top_k))
            mrr_scores.append(mrr_at_k(recommendations, [target_item], self.top_k))
            ndcg_scores.append(ndcg_at_k(recommendations, [target_item], self.top_k))

        count = len(hr_scores)
        return {
            "hr@k": sum(hr_scores) / count,
            "mrr@k": sum(mrr_scores) / count,
            "ndcg@k": sum(ndcg_scores) / count,
        }

    def log_metrics(self, metrics: dict[str, float]) -> None:
        return None
