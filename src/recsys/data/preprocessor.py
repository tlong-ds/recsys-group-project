"""Data preprocessor: cleans and transforms raw DataFrames."""

from __future__ import annotations

import pandas as pd


class Preprocessor:
    """Clean and encode raw interaction data."""

    def fit(self, interactions: pd.DataFrame) -> "Preprocessor":
        """Fit any encoders/scalers on the training data."""
        # TODO: implement
        raise NotImplementedError

    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations."""
        # TODO: implement
        raise NotImplementedError

    def fit_transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(interactions).transform(interactions)
