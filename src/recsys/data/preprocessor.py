"""Data preprocessing for session-based interactions."""

from __future__ import annotations

import pandas as pd


class SessionPreprocessor:
    """Clean, type-cast, and sort interaction data."""

    def __init__(
        self,
        session_col: str = "session_id",
        item_col: str = "item_id",
        timestamp_col: str = "timestamp",
    ) -> None:
        self.session_col = session_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col

    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Return cleaned interactions ordered within each session."""
        columns = [self.session_col, self.item_col, self.timestamp_col]
        missing = [column for column in columns if column not in interactions.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        cleaned = interactions.copy()
        cleaned = cleaned.dropna(subset=columns)
        cleaned[self.timestamp_col] = pd.to_datetime(cleaned[self.timestamp_col], utc=True)
        cleaned = cleaned.sort_values([self.session_col, self.timestamp_col]).reset_index(
            drop=True
        )
        cleaned[self.item_col] = cleaned[self.item_col].astype(int)
        return cleaned
