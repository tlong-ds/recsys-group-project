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
                "eventdate": [
                    "2024-01-01T10:00:01",
                    "2024-01-01T10:00:00",
                    "2024-01-01T10:01:00",
                ],
            }
        )

        processed = SessionPreprocessor().transform(interactions)

        self.assertEqual(processed["item_id"].tolist(), [10, 20, 30])
        self.assertTrue(pd.api.types.is_datetime64tz_dtype(processed["eventdate"]))

    def test_filter_sessions_keeps_last_max_length_items(self) -> None:
        interactions = pd.DataFrame(
            {
                "session_id": [1, 1, 1, 1, 2, 2],
                "item_id": [10, 11, 12, 13, 20, 21],
                "eventdate": pd.to_datetime(
                    [
                        "2024-01-01T10:00:00Z",
                        "2024-01-01T10:01:00Z",
                        "2024-01-01T10:02:00Z",
                        "2024-01-01T10:03:00Z",
                        "2024-01-01T10:00:00Z",
                        "2024-01-01T10:01:00Z",
                    ],
                    utc=True,
                ),
            }
        )

        filtered = SessionPreprocessor().filter_sessions(
            interactions,
            min_length=2,
            max_length=3,
        )

        session_1_items = filtered[filtered["session_id"] == 1]["item_id"].tolist()
        session_2_items = filtered[filtered["session_id"] == 2]["item_id"].tolist()

        self.assertEqual(session_1_items, [11, 12, 13])
        self.assertEqual(session_2_items, [20, 21])

    def test_filter_sessions_raises_for_non_positive_max_length(self) -> None:
        interactions = pd.DataFrame(
            {
                "session_id": [1, 1],
                "item_id": [10, 11],
                "eventdate": pd.to_datetime(
                    ["2024-01-01T10:00:00Z", "2024-01-01T10:01:00Z"],
                    utc=True,
                ),
            }
        )

        with self.assertRaises(ValueError):
            SessionPreprocessor().filter_sessions(interactions, min_length=1, max_length=0)
