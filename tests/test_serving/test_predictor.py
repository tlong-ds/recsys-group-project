"""Tests for Predictor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recsys.models.srgnn import SRGNNRecommender
from recsys.serving.predictor import Predictor


class TestPredictor(unittest.TestCase):
    def test_predictor_loads_model_and_returns_recommendations(self) -> None:
        examples = pd.DataFrame(
            {
                "session_id": [1, 1],
                "context_items": [[1], [1, 2]],
                "target_item": [2, 3],
                "last_item_id": [1, 2],
                "context_length": [1, 2],
            }
        )
        model = SRGNNRecommender().fit(examples)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = model.save(Path(tmp_dir))
            predictor = Predictor.from_path(str(artifact_path))

        self.assertEqual(predictor.get_recommendations([1, 2], top_k=1), [3])
