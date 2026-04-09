"""Dataset splitter: train / validation / test splits."""

from __future__ import annotations

import pandas as pd


def split_by_time(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split into train, validation, and test sets."""
    # TODO: implement
    raise NotImplementedError


def split_random(
    df: pd.DataFrame,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random split into train, validation, and test sets."""
    # TODO: implement
    raise NotImplementedError
