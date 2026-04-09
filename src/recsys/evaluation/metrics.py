"""Ranking and rating metrics for recommendation evaluation."""

from __future__ import annotations

import numpy as np


def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    """Fraction of top-k recommendations that are relevant."""
    # TODO: implement
    raise NotImplementedError


def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    """Fraction of relevant items found in top-k recommendations."""
    # TODO: implement
    raise NotImplementedError


def ndcg_at_k(recommended: list, relevant: list, k: int) -> float:
    """Normalised discounted cumulative gain at k."""
    # TODO: implement
    raise NotImplementedError


def mean_average_precision(
    all_recommended: list[list],
    all_relevant: list[list],
) -> float:
    """Mean average precision across all users."""
    # TODO: implement
    raise NotImplementedError


def hit_rate_at_k(
    all_recommended: list[list],
    all_relevant: list[list],
    k: int,
) -> float:
    """Fraction of users for whom at least one relevant item is in top-k."""
    # TODO: implement
    raise NotImplementedError
