"""Inference wrapper for the trained session recommender."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Predictor:
    """Load a model artifact and serve next-item recommendations."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @classmethod
    def from_path(cls, model_path: str) -> Predictor:
        """Load a persisted model artifact from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        from recsys.models.srgnn import SRGNNRecommender

        return cls(SRGNNRecommender.load(path))

    @classmethod
    def from_model_registry(
        cls,
        *,
        mlflow_config: dict[str, Any],
        model_name: str,
        model_alias: str | None = None,
        model_version: str | None = None,
        artifact_path: str = "registered_model",
    ) -> tuple[Predictor, dict[str, str]]:
        """Load model metadata from MLflow Registry, then fetch serving artifact."""
        from recsys.training.tracking import configure_tracking

        configure_tracking({"mlflow": mlflow_config})
        client = _mlflow_client()

        if model_alias:
            version = client.get_model_version_by_alias(model_name, model_alias)
        elif model_version:
            version = client.get_model_version(model_name, str(model_version))
        else:
            raise ValueError("Either model_alias or model_version must be provided.")

        local_path = Path(client.download_artifacts(version.run_id, artifact_path))
        from recsys.models.srgnn import SRGNNRecommender

        predictor = cls(SRGNNRecommender.load(local_path))
        metadata = {
            "source": "mlflow_registry",
            "model_name": model_name,
            "model_version": str(version.version),
            "model_alias": model_alias or "",
            "run_id": str(version.run_id),
            "artifact_path": str(local_path),
        }
        return predictor, metadata

    def get_recommendations(
        self, item_sequence: list[int], top_k: int = 10
    ) -> list[int]:
        """Return the top-k next-item predictions for a session context."""
        return self.model.recommend(item_sequence, top_k=top_k)

    def get_scores(self, item_sequence: list[int], item_ids: list[int]) -> list[float]:
        """Return scores for candidate items."""
        return self.model.score(item_sequence, item_ids)


def _mlflow_client():
    from mlflow.tracking import MlflowClient

    return MlflowClient()
