"""Tests for evaluation metrics."""

import pytest

from recsys.evaluation.metrics import (
    hit_rate_at_k,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestMetrics:
    def test_precision_at_k_not_implemented(self):
        with pytest.raises(NotImplementedError):
            precision_at_k(recommended=[1, 2, 3], relevant=[1, 2], k=3)

    def test_recall_at_k_not_implemented(self):
        with pytest.raises(NotImplementedError):
            recall_at_k(recommended=[1, 2, 3], relevant=[1, 2], k=3)

    def test_ndcg_at_k_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ndcg_at_k(recommended=[1, 2, 3], relevant=[1, 2], k=3)

    def test_map_not_implemented(self):
        with pytest.raises(NotImplementedError):
            mean_average_precision(all_recommended=[[1, 2]], all_relevant=[[1]])

    def test_hit_rate_not_implemented(self):
        with pytest.raises(NotImplementedError):
            hit_rate_at_k(all_recommended=[[1, 2]], all_relevant=[[1]], k=2)
