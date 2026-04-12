"""Tests for SRGNNRecommender."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recsys.models.srgnn import SRGNNRecommender


def _training_examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": [1, 1, 2, 2],
            "context_items": [[1], [1, 2], [4], [4, 2]],
            "target_item": [2, 3, 2, 5],
            "last_item_id": [1, 2, 4, 2],
            "context_length": [1, 2, 1, 2],
        }
    )


class TestSRGNNRecommender(unittest.TestCase):
    def test_recommend_uses_transition_scores(self) -> None:
        model = SRGNNRecommender().fit(_training_examples())

        recommendations = model.recommend([1, 2], top_k=2)

        self.assertEqual(recommendations[0], 3)

    def test_save_and_load_round_trip(self) -> None:
        model = SRGNNRecommender().fit(_training_examples())

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = model.save(Path(tmp_dir))
            restored = SRGNNRecommender.load(artifact_path)

        self.assertEqual(restored.recommend([1, 2], top_k=2)[0], 3)
