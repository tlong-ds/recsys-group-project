"""Feature engineering: builds user and item feature matrices."""

from __future__ import annotations

import pandas as pd


class FeatureEngineer:
    """Compute interaction-based and content-based features."""

    def build_user_features(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Derive per-user aggregate features from interaction history."""
        # TODO: implement
        raise NotImplementedError

    def build_item_features(
        self,
        interactions: pd.DataFrame,
        item_metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine interaction statistics with item content features."""
        # TODO: implement
        raise NotImplementedError
