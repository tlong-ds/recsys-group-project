"""Tests for temporal dataset splitting."""

from __future__ import annotations

import unittest

import pandas as pd

from recsys.data.splitter import split_by_time


class TestSplitter(unittest.TestCase):
    def test_split_by_time_preserves_temporal_order(self) -> None:
        interactions = pd.DataFrame(
            {
                "session_id": [1, 1, 2, 2, 3],
                "item_id": [1, 2, 3, 4, 5],
                "eventdate": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-05",
                    ],
                    utc=True,
                ),
            }
        )

        train_df, val_df, test_df = split_by_time(interactions, val_ratio=0.2, test_ratio=0.2)

        self.assertLess(train_df["eventdate"].max(), val_df["eventdate"].min())
        self.assertLess(val_df["eventdate"].max(), test_df["eventdate"].min())
