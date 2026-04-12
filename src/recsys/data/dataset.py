"""Dataset builders for train/validation/test session splits."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from recsys.data.splitter import split_by_time


@dataclass
class DatasetSplit:
    """Materialised train, validation, and test tables."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class DatasetBuilder:
    """Build dataset splits from preprocessed interactions."""

    def __init__(self, timestamp_col: str = "timestamp") -> None:
        self.timestamp_col = timestamp_col

    def build_splits(
        self,
        interactions: pd.DataFrame,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> DatasetSplit:
        train_df, val_df, test_df = split_by_time(
            interactions,
            timestamp_col=self.timestamp_col,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        return DatasetSplit(train=train_df, validation=val_df, test=test_df)
