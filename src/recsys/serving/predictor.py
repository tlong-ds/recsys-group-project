"""Predictor: wraps a trained model for online/batch inference."""

from __future__ import annotations

from recsys.models.base_model import BaseRecsysModel


class Predictor:
    """Load a trained model and serve recommendations."""

    def __init__(self, model: BaseRecsysModel) -> None:
        self.model = model

    @classmethod
    def from_path(cls, model_path: str) -> "Predictor":
        """Load a persisted model and return a ready Predictor."""
        # TODO: implement
        raise NotImplementedError

    def get_recommendations(self, user_id: int, top_k: int = 10) -> list[int]:
        """Return top-k item IDs for the given user."""
        # TODO: implement
        raise NotImplementedError

    def get_scores(self, user_id: int, item_ids: list[int]) -> list[float]:
        """Return predicted scores for specific (user, item) pairs."""
        # TODO: implement
        raise NotImplementedError
