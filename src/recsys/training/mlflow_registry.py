"""MLflow Model Registry helpers for training and serving."""

from __future__ import annotations

from typing import Any


def register_model_version(
    *,
    config: dict[str, Any],
    run_id: str,
    source_artifact_path: str = "model_core",
) -> dict[str, Any] | None:
    """Register a model version in MLflow Model Registry when enabled."""
    registry_cfg = _registry_config(config)
    if not _as_bool(registry_cfg.get("enabled", False)):
        return None

    model_name = str(registry_cfg.get("model_name", "recsys-srgnn"))
    alias = registry_cfg.get("register_alias")
    source = f"runs:/{run_id}/{source_artifact_path}"

    client = _mlflow_client()
    try:
        client.create_registered_model(model_name)
    except Exception:
        # Model may already exist in concurrent environments.
        pass
    version = client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
    )
    if alias:
        client.set_registered_model_alias(
            name=model_name,
            alias=str(alias),
            version=str(version.version),
        )
    return {
        "model_name": model_name,
        "model_version": str(version.version),
        "alias": str(alias) if alias else None,
        "run_id": run_id,
        "source": source,
    }


def _registry_config(config: dict[str, Any]) -> dict[str, Any]:
    mlflow_cfg = config.get("mlflow", {})
    if not isinstance(mlflow_cfg, dict):
        return {}
    registry_cfg = mlflow_cfg.get("registry", {})
    return registry_cfg if isinstance(registry_cfg, dict) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _mlflow_client():
    from mlflow.tracking import MlflowClient

    return MlflowClient()
