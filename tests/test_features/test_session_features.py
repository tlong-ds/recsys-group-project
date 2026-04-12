"""Tests for SessionFeatureBuilder."""

from __future__ import annotations

import unittest

import pandas as pd

from recsys.features.session_features import SessionFeatureBuilder


class TestSessionFeatureBuilder(unittest.TestCase):
    def test_build_examples_creates_prefix_target_pairs(self) -> None:
        interactions = pd.DataFrame(
            {
                "session_id": [1, 1, 1],
                "item_id": [10, 20, 30],
                "timestamp": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True
                ),
            }
        )

        examples = SessionFeatureBuilder().build_examples(interactions)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples.iloc[0]["context_items"], [10])
        self.assertEqual(examples.iloc[1]["target_item"], 30)
