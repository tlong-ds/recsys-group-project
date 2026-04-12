"""Inference wrapper for the trained session recommender."""

from __future__ import annotations

from pathlib import Path

from recsys.models.srgnn import SRGNNRecommender


class Predictor:
    """Load a model artifact and serve next-item recommendations."""

    def __init__(self, model: SRGNNRecommender) -> None:
        self.model = model

    @classmethod
    def from_path(cls, model_path: str) -> "Predictor":
        """Load a persisted model artifact from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        return cls(SRGNNRecommender.load(path))

    def get_recommendations(self, item_sequence: list[int], top_k: int = 10) -> list[int]:
        """Return the top-k next-item predictions for a session context."""
        return self.model.recommend(item_sequence, top_k=top_k)

    def get_scores(self, item_sequence: list[int], item_ids: list[int]) -> list[float]:
        """Return scores for candidate items."""
        return self.model.score(item_sequence, item_ids)
