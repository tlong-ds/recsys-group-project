"""Tests for the local model registry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recsys.models.srgnn import SRGNNRecommender
from recsys.training.registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):
    def test_register_writes_latest_artifact(self) -> None:
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
            registry = ModelRegistry(Path(tmp_dir))
            artifact_path = registry.register(model, config={"model": {"name": "srgnn"}}, metrics={})

            self.assertTrue(artifact_path.exists())
            self.assertTrue(registry.latest_model_path().exists())
