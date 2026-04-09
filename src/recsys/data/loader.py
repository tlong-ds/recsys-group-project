"""Data loader: reads raw interaction and item/user metadata files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoader:
    """Load raw datasets from disk."""

    def __init__(self, raw_path: str | Path) -> None:
        self.raw_path = Path(raw_path)

    def load_interactions(self) -> pd.DataFrame:
        """Return a DataFrame of (user_id, item_id, rating, timestamp)."""
        # TODO: implement
        raise NotImplementedError

    def load_items(self) -> pd.DataFrame:
        """Return a DataFrame of item metadata."""
        # TODO: implement
        raise NotImplementedError

    def load_users(self) -> pd.DataFrame:
        """Return a DataFrame of user metadata."""
        # TODO: implement
        raise NotImplementedError
