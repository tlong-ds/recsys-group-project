"""Temporal splitting for session-based training data."""

from __future__ import annotations

import pandas as pd


def split_by_time(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split a DataFrame into train, validation, and test sets."""
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("Split ratios must be in the [0, 1) range")
    if val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must sum to less than 1")
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    n_rows = len(ordered)
    test_start = int(n_rows * (1 - test_ratio))
    val_start = int(n_rows * (1 - test_ratio - val_ratio))

    train_df = ordered.iloc[:val_start].reset_index(drop=True)
    val_df = ordered.iloc[val_start:test_start].reset_index(drop=True)
    test_df = ordered.iloc[test_start:].reset_index(drop=True)
    return train_df, val_df, test_df
