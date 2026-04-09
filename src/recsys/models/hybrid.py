"""Hybrid recommender combining CF and content-based scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.models.base_model import BaseRecsysModel
from recsys.models.collaborative_filtering import CollaborativeFilteringModel
from recsys.models.content_based import ContentBasedModel


class HybridModel(BaseRecsysModel):
    """Weighted combination of CF and content-based models (placeholder)."""

    def __init__(
        self,
        cf_weight: float = 0.6,
        cb_weight: float = 0.4,
    ) -> None:
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_model = CollaborativeFilteringModel()
        self.cb_model = ContentBasedModel()

    def fit(self, interactions: pd.DataFrame) -> "HybridModel":
        # TODO: fit both sub-models
        raise NotImplementedError

    def predict(self, user_ids: list, item_ids: list) -> np.ndarray:
        # TODO: weighted combination of CF and CB scores
        raise NotImplementedError

    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        # TODO: merge ranked lists and return top-k
        raise NotImplementedError
