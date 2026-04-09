"""Content-based filtering model using item embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.models.base_model import BaseRecsysModel


class ContentBasedModel(BaseRecsysModel):
    """Cosine-similarity content-based recommender (placeholder)."""

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = embedding_dim

    def fit(self, interactions: pd.DataFrame) -> "ContentBasedModel":
        # TODO: build item embeddings from content features
        raise NotImplementedError

    def predict(self, user_ids: list, item_ids: list) -> np.ndarray:
        # TODO: implement similarity scoring
        raise NotImplementedError

    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        # TODO: implement top-k retrieval by similarity
        raise NotImplementedError
