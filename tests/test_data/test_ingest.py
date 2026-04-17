"""Tests for DataLoader."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recsys.data.ingest import DataLoader


class TestDataLoader(unittest.TestCase):
    def test_load_interactions_reads_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir)
            pd.DataFrame(
                {
                    "session_id": [1, 1],
                    "item_id": [10, 11],
                    "timestamp": ["2024-01-01", "2024-01-02"],
                }
            ).to_csv(raw_path / "interactions.csv", index=False)

            loader = DataLoader(raw_path=raw_path)
            interactions = loader.load_interactions()

        self.assertEqual(list(interactions.columns), ["session_id", "item_id", "timestamp"])
        self.assertEqual(len(interactions), 2)
 