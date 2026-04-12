"""Tests for SessionPreprocessor."""

from __future__ import annotations

import unittest

import pandas as pd

from recsys.data.preprocessor import SessionPreprocessor


class TestSessionPreprocessor(unittest.TestCase):
    def test_transform_sorts_rows_within_session(self) -> None:
        interactions = pd.DataFrame(
            {
                "session_id": [1, 1, 2],
                "item_id": [20, 10, 30],
                "timestamp": [
                    "2024-01-01T10:00:01",
                    "2024-01-01T10:00:00",
                    "2024-01-01T10:01:00",
                ],
            }
        )

        processed = SessionPreprocessor().transform(interactions)

        self.assertEqual(processed["item_id"].tolist(), [10, 20, 30])
        self.assertTrue(pd.api.types.is_datetime64tz_dtype(processed["timestamp"]))
