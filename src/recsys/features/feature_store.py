"""Feature store: persist and retrieve computed feature sets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class FeatureStore:
    """Simple file-backed feature store."""

    def __init__(self, features_path: str | Path) -> None:
        self.features_path = Path(features_path)

    def save(self, name: str, df: pd.DataFrame) -> None:
        """Persist a feature DataFrame to the store."""
        # TODO: implement
        raise NotImplementedError

    def load(self, name: str) -> pd.DataFrame:
        """Retrieve a feature DataFrame from the store."""
        # TODO: implement
        raise NotImplementedError

    def list_features(self) -> list[str]:
        """Return names of all stored feature sets."""
        # TODO: implement
        raise NotImplementedError
