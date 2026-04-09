"""Matrix-factorisation collaborative filtering model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.models.base_model import BaseRecsysModel


class CollaborativeFilteringModel(BaseRecsysModel):
    """Matrix factorisation via SGD (placeholder)."""

    def __init__(
        self,
        n_factors: int = 64,
        n_epochs: int = 20,
        lr: float = 0.001,
        reg: float = 0.01,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

    def fit(self, interactions: pd.DataFrame) -> "CollaborativeFilteringModel":
        # TODO: implement matrix factorisation training
        raise NotImplementedError

    def predict(self, user_ids: list, item_ids: list) -> np.ndarray:
        # TODO: implement dot-product scoring
        raise NotImplementedError

    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        # TODO: implement top-k retrieval
        raise NotImplementedError
