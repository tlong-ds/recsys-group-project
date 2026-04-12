"""Tests for ranking metrics."""

from __future__ import annotations

import unittest

from recsys.evaluation.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k


class TestMetrics(unittest.TestCase):
    def test_hit_rate_at_k_returns_one_when_target_is_present(self) -> None:
        self.assertEqual(hit_rate_at_k([4, 5, 6], [5], 3), 1.0)

    def test_mrr_at_k_uses_first_relevant_rank(self) -> None:
        self.assertAlmostEqual(mrr_at_k([4, 5, 6], [5], 3), 0.5)

    def test_ndcg_at_k_is_normalised(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([5, 4, 6], [5], 3), 1.0)
