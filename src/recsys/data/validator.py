"""Validation rules for raw recommender datasets."""

from __future__ import annotations

import pandas as pd


class DataValidator:
    """Validate required columns and basic data quality constraints."""

    def __init__(
        self,
        session_col: str = "session_id",
        item_col: str = "item_id",
        timestamp_col: str = "timestamp",
    ) -> None:
        self.session_col = session_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col

    def validate_interactions(self, interactions: pd.DataFrame) -> None:
        required = {self.session_col, self.item_col, self.timestamp_col}
        missing = required.difference(interactions.columns)
        if missing:
            raise ValueError(f"Missing interaction columns: {sorted(missing)}")
        if interactions.empty:
            raise ValueError("Interaction dataset is empty")

    def validate_items(self, items: pd.DataFrame, item_col: str = "item_id") -> None:
        if items.empty:
            return
        if item_col not in items.columns:
            raise ValueError(f"Missing item column: {item_col}")
