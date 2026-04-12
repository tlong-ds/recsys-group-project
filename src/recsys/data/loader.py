"""Data loading utilities for raw recommender datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoader:
    """Load interactions and optional metadata tables from disk."""

    def __init__(
        self,
        raw_path: str | Path,
        interactions_file: str = "interactions.csv",
        items_file: str = "items.csv",
        users_file: str = "users.csv",
    ) -> None:
        self.raw_path = Path(raw_path)
        self.interactions_file = interactions_file
        self.items_file = items_file
        self.users_file = users_file

    def _load_csv(self, filename: str, required: bool = True) -> pd.DataFrame:
        path = self.raw_path / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Expected dataset file at {path}")
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_interactions(self) -> pd.DataFrame:
        """Return interaction rows used to build sessions."""
        return self._load_csv(self.interactions_file, required=True)

    def load_items(self, required: bool = False) -> pd.DataFrame:
        """Return optional item metadata."""
        return self._load_csv(self.items_file, required=required)

    def load_users(self, required: bool = False) -> pd.DataFrame:
        """Return optional user metadata."""
        return self._load_csv(self.users_file, required=required)
