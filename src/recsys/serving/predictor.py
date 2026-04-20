"""Inference wrapper for the trained session recommender."""

from __future__ import annotations

import hashlib
import shutil
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
        cache_dir: str | None = None,
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

        local_path, cache_hit = _resolve_registry_artifact(
            client=client,
            run_id=str(version.run_id),
            artifact_path=artifact_path,
            cache_dir=cache_dir,
        )
        from recsys.models.srgnn import SRGNNRecommender

        predictor = cls(SRGNNRecommender.load(local_path))
        metadata = {
            "source": "mlflow_registry",
            "model_name": model_name,
            "model_version": str(version.version),
            "model_alias": model_alias or "",
            "run_id": str(version.run_id),
            "artifact_path": str(local_path),
            "cache_hit": "true" if cache_hit else "false",
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


def _resolve_registry_artifact(
    *,
    client: Any,
    run_id: str,
    artifact_path: str,
    cache_dir: str | None,
) -> tuple[Path, bool]:
    if not cache_dir:
        return Path(client.download_artifacts(run_id, artifact_path)), False

    cache_root = Path(cache_dir).expanduser()
    key_input = f"{run_id}:{artifact_path}".encode()
    artifact_key = hashlib.sha1(key_input).hexdigest()[:16]
    target = cache_root / run_id / artifact_key

    if _looks_like_model_artifact(target):
        return target, True

    downloaded = Path(client.download_artifacts(run_id, artifact_path))
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    if downloaded.is_dir():
        shutil.copytree(downloaded, target, dirs_exist_ok=True)
    else:
        target.write_bytes(downloaded.read_bytes())

    return target, False


def _looks_like_model_artifact(path: Path) -> bool:
    if path.is_file():
        return True
    return (path / "model.json").exists() or (path / "model.pt").exists()
