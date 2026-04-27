"""Warm and validate the shared registry model cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from loguru import logger
from mlflow.tracking import MlflowClient

from recsys.serving.model_provider import resolve_model_cache_dir
from recsys.serving.predictor import (
    Predictor,
    _looks_like_model_artifact,
    resolve_registry_cache_path,
)
from recsys.training.tracking import configure_tracking
from recsys.utils.config import load_config

_DEFAULT_CONFIG_PATH = Path("/app/configs/serving_config.yaml")


def _resolve_registry_target(
    *,
    mlflow_config: dict[str, Any],
    registry_config: dict[str, Any],
) -> tuple[str, str, str, str]:
    model_name = str(
        os.getenv("RECSYS_DEPLOY_MODEL_NAME")
        or registry_config.get("model_name", "recsys-serving")
    )
    pinned_version = os.getenv("RECSYS_DEPLOY_MODEL_VERSION")
    pinned_run_id = os.getenv("RECSYS_DEPLOY_RUN_ID")
    model_alias = str(registry_config.get("model_alias", "Production"))

    configure_tracking({"mlflow": mlflow_config})
    client = MlflowClient()

    if pinned_version:
        version = client.get_model_version(model_name, str(pinned_version))
    else:
        version = client.get_model_version_by_alias(model_name, model_alias)

    resolved_version = str(version.version)
    resolved_run_id = str(pinned_run_id or version.run_id)
    if pinned_run_id and str(version.run_id) != str(pinned_run_id):
        raise ValueError(
            "Pinned run_id does not match the configured model version in MLflow. "
            f"Expected {version.run_id}, got {pinned_run_id}."
        )

    return model_name, resolved_version, resolved_run_id, model_alias


def run_warmup(config_path: Path) -> dict[str, str]:
    config = load_config(config_path)
    serving_cfg = config.get("serving", {})
    mlflow_cfg = config.get("mlflow", {})
    registry_cfg = serving_cfg.get("model_registry", {})

    if not isinstance(registry_cfg, dict) or not bool(
        registry_cfg.get("enabled", False)
    ):
        return {"status": "skipped", "reason": "model registry disabled"}

    model_name, model_version, run_id, model_alias = _resolve_registry_target(
        mlflow_config=mlflow_cfg,
        registry_config=registry_cfg,
    )

    artifact_path = str(registry_cfg.get("artifact_path", "registered_model"))
    cache_root = resolve_model_cache_dir(registry_cfg) or "/app/models/cache"

    cache_target = resolve_registry_cache_path(
        cache_root=cache_root,
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        artifact_path=artifact_path,
    )
    ready_marker = cache_target / ".ready"

    if ready_marker.exists() and _looks_like_model_artifact(cache_target):
        return {
            "status": "hit",
            "model_name": model_name,
            "model_version": model_version,
            "run_id": run_id,
            "artifact_path": str(cache_target),
        }

    _, metadata = Predictor.from_model_registry(
        mlflow_config=mlflow_cfg,
        model_name=model_name,
        model_alias=None,
        model_version=model_version,
        run_id=run_id,
        artifact_path=artifact_path,
        cache_dir=cache_root,
    )
    metadata["status"] = "downloaded"
    metadata["requested_alias"] = model_alias
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm recsys model cache")
    parser.add_argument(
        "--config",
        default=os.getenv("RECSYS_SERVING_CONFIG", str(_DEFAULT_CONFIG_PATH)),
        help="Path to serving config file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when warmup fails",
    )
    args = parser.parse_args()

    try:
        result = run_warmup(Path(args.config))
        logger.info("Model cache warmup result: {}", result)
    except Exception as exc:
        logger.error("Model cache warmup failed: {}", exc)
        if args.strict:
            raise


if __name__ == "__main__":
    main()
