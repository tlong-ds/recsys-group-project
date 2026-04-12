"""Offline ranking metrics for next-item prediction."""

from __future__ import annotations

import math


def hit_rate_at_k(recommended: list[int], relevant: list[int], k: int) -> float:
    """Return 1 when any relevant item appears in the top-k list, else 0."""
    top_k = recommended[:k]
    return float(any(item in relevant for item in top_k))


def mrr_at_k(recommended: list[int], relevant: list[int], k: int) -> float:
    """Return reciprocal rank of the first relevant item within top-k."""
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(recommended: list[int], relevant: list[int], k: int) -> float:
    """Return normalised discounted cumulative gain for binary relevance."""
    gains = [
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(recommended[:k], start=1)
        if item in relevant
    ]
    dcg = sum(gains)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg
