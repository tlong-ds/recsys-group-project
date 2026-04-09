"""Abstract base class for all recommendation models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseRecsysModel(ABC):
    """Common interface for recsys models."""

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> "BaseRecsysModel":
        """Train the model on interaction data."""
        ...

    @abstractmethod
    def predict(self, user_ids: list, item_ids: list) -> np.ndarray:
        """Return predicted scores for (user, item) pairs."""
        ...

    @abstractmethod
    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        """Return top-k item IDs for the given user."""
        ...

    def save(self, path: str) -> None:
        """Persist model artefacts to disk."""
        # TODO: implement
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BaseRecsysModel":
        """Load model artefacts from disk."""
        # TODO: implement
        raise NotImplementedError
