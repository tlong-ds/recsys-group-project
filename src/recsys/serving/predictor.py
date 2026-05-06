"""Inference wrapper for the trained session recommender."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

_DOWNLOAD_LOCK_FILENAME = ".downloading"
_READY_MARKER_FILENAME = ".ready"
_CACHE_LOCK_WAIT_SECONDS = 300
_CACHE_LOCK_STALE_SECONDS = 600
_CACHE_LOCK_POLL_SECONDS = 1.0


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
        return cls(_load_model_artifact(path))

    @classmethod
    def from_model_registry(
        cls,
        *,
        mlflow_config: dict[str, Any],
        model_name: str,
        model_alias: str | None = None,
        model_version: str | None = None,
        run_id: str | None = None,
        artifact_path: str = "registered_model",
        cache_dir: str | None = None,
    ) -> tuple[Predictor, dict[str, str]]:
        """Load model metadata from MLflow Registry, then fetch serving artifact."""
        from recsys.training.tracking import configure_tracking

        configure_tracking({"mlflow": mlflow_config})
        client = _mlflow_client()

        if model_version:
            version = client.get_model_version(model_name, str(model_version))
            resolved_alias = ""
        elif model_alias:
            version = client.get_model_version_by_alias(model_name, model_alias)
            resolved_alias = str(model_alias)
        else:
            raise ValueError(
                "Either model_alias or model_version must be provided."
            )

        resolved_version = str(version.version)
        registry_run_id = str(version.run_id)
        resolved_run_id = str(run_id or registry_run_id)
        if run_id and str(run_id) != registry_run_id:
            raise ValueError(
                "Provided run_id does not match the MLflow Registry model version. "
                f"Expected {registry_run_id}, got {run_id}."
            )

        local_path, cache_hit = _resolve_registry_artifact(
            client=client,
            model_name=model_name,
            model_version=resolved_version,
            run_id=resolved_run_id,
            artifact_path=artifact_path,
            cache_dir=cache_dir,
        )
        predictor = cls(_load_model_artifact(local_path))
        metadata = {
            "source": "mlflow_registry",
            "model_name": model_name,
            "model_version": resolved_version,
            "model_alias": resolved_alias,
            "run_id": resolved_run_id,
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

    def known_item_count(self) -> int:
        """Return the known catalog size exposed by the loaded model."""
        item_to_idx = getattr(self.model, "_item_to_idx", None)
        if isinstance(item_to_idx, dict) and item_to_idx:
            return len(item_to_idx)

        n_items = getattr(self.model, "n_items", 0)
        try:
            return max(0, int(n_items))
        except (TypeError, ValueError):
            return 0

    def count_unknown_items(self, item_sequence: list[int]) -> int:
        """Count request items not known by the loaded model catalog."""
        return sum(1 for item in item_sequence if not self._is_known_item(item))

    def input_quality(self, item_sequence: list[int]) -> dict[str, int | float]:
        """Return lightweight input-quality signals for online monitoring."""
        sequence_length = len(item_sequence)
        unknown_items = self.count_unknown_items(item_sequence)
        known_items = sequence_length - unknown_items
        oov_ratio = unknown_items / sequence_length if sequence_length else 0.0
        return {
            "sequence_length": sequence_length,
            "known_items": known_items,
            "unknown_items": unknown_items,
            "oov_ratio": oov_ratio,
            "known_catalog_items": self.known_item_count(),
        }

    def _is_known_item(self, item: int) -> bool:
        item_to_idx = getattr(self.model, "_item_to_idx", None)
        if isinstance(item_to_idx, dict) and item_to_idx:
            return int(item) in item_to_idx

        n_items = getattr(self.model, "n_items", 0)
        try:
            return 1 <= int(item) <= int(n_items)
        except (TypeError, ValueError):
            return False


def resolve_registry_cache_path(
    *,
    cache_root: str | Path,
    model_name: str,
    model_version: str,
    run_id: str,
    artifact_path: str,
) -> Path:
    """Build a deterministic cache location for a concrete registry artifact."""
    root = Path(cache_root).expanduser()
    artifact_key = hashlib.sha256(f"{run_id}:{artifact_path}".encode()).hexdigest()[:16]
    return (
        root
        / _sanitize_cache_segment(model_name)
        / _sanitize_cache_segment(model_version)
        / _sanitize_cache_segment(run_id)
        / artifact_key
    )


def _sanitize_cache_segment(raw: str) -> str:
    allowed = {"-", "_", "."}
    cleaned = "".join(
        ch if ch.isalnum() or ch in allowed else "_" for ch in str(raw).strip()
    )
    return cleaned or "unknown"


def _mlflow_client():
    from mlflow.tracking import MlflowClient

    return MlflowClient()


def _resolve_registry_artifact(
    *,
    client: Any,
    model_name: str,
    model_version: str,
    run_id: str,
    artifact_path: str,
    cache_dir: str | None,
) -> tuple[Path, bool]:
    if not cache_dir:
        return Path(client.download_artifacts(run_id, artifact_path)), False

    target = resolve_registry_cache_path(
        cache_root=cache_dir,
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        artifact_path=artifact_path,
    )
    lock_marker = target / _DOWNLOAD_LOCK_FILENAME
    ready_marker = target / _READY_MARKER_FILENAME

    if _cache_is_ready(target, ready_marker):
        return target, True

    wait_started = time.monotonic()
    while True:
        target.mkdir(parents=True, exist_ok=True)

        if _cache_is_ready(target, ready_marker):
            return target, True

        if _acquire_download_lock(lock_marker):
            try:
                ready_marker.unlink(missing_ok=True)
                downloaded = Path(client.download_artifacts(run_id, artifact_path))
                _sync_downloaded_artifact(target=target, downloaded=downloaded)
                ready_marker.write_text(
                    f"run_id={run_id}\nartifact_path={artifact_path}\n",
                    encoding="utf-8",
                )
                return target, False
            finally:
                _release_download_lock(lock_marker)

        if _cache_is_ready(target, ready_marker):
            return target, True

        if _lock_is_stale(lock_marker):
            _release_download_lock(lock_marker)
            continue

        if time.monotonic() - wait_started > _CACHE_LOCK_WAIT_SECONDS:
            # Last-resort progress path if lock ownership cannot be observed.
            downloaded = Path(client.download_artifacts(run_id, artifact_path))
            _sync_downloaded_artifact(target=target, downloaded=downloaded)
            ready_marker.write_text(
                f"run_id={run_id}\nartifact_path={artifact_path}\n",
                encoding="utf-8",
            )
            return target, False

        time.sleep(_CACHE_LOCK_POLL_SECONDS)


def _cache_is_ready(target: Path, ready_marker: Path) -> bool:
    return ready_marker.exists() and _looks_like_model_artifact(target)


def _acquire_download_lock(lock_marker: Path) -> bool:
    lock_marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\ntime={time.time()}\n")
    return True


def _release_download_lock(lock_marker: Path) -> None:
    lock_marker.unlink(missing_ok=True)


def _lock_is_stale(lock_marker: Path) -> bool:
    if not lock_marker.exists():
        return False
    age = time.time() - lock_marker.stat().st_mtime
    return age >= _CACHE_LOCK_STALE_SECONDS


def _sync_downloaded_artifact(*, target: Path, downloaded: Path) -> None:
    _clear_cache_payload(target)
    if downloaded.is_dir():
        for item in downloaded.iterdir():
            destination = target / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)
        return

    shutil.copy2(downloaded, target / downloaded.name)


def _clear_cache_payload(target: Path) -> None:
    keep_names = {_DOWNLOAD_LOCK_FILENAME, _READY_MARKER_FILENAME}
    for item in target.iterdir():
        if item.name in keep_names:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _looks_like_model_artifact(path: Path) -> bool:
    if path.is_file():
        return True
    if not path.exists():
        return False
    return (path / "model.json").exists() or (path / "model.pt").exists()


def _load_model_artifact(path: Path) -> Any:
    model_type = _model_type_from_artifact(path)
    if model_type == "tagnn":
        from recsys.models.tagnn import TAGNNRecommender

        return TAGNNRecommender.load(path)
    if model_type == "ggnn":
        from recsys.models.ggnn import GGNNRecommender

        return GGNNRecommender.load(path)
    if model_type == "srgnn":
        from recsys.models.srgnn import SRGNNRecommender

        return SRGNNRecommender.load(path)
    raise ValueError(
        f"Unsupported model_type {model_type!r} in model metadata. "
        "Expected one of: 'srgnn', 'tagnn', 'ggnn'."
    )


def _model_type_from_artifact(path: Path) -> str:
    artifact_dir = path if path.is_dir() else path.parent
    meta_path = artifact_dir / "model.json"
    if not meta_path.exists():
        return "srgnn"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    model_type = str(metadata.get("model_type", "srgnn")).strip().lower()
    return model_type or "srgnn"
