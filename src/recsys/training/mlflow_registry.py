"""MLflow Model Registry helpers for training and serving."""

from __future__ import annotations

import re
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

    model_name = _resolve_registry_model_name(config=config, registry_cfg=registry_cfg)
    alias = registry_cfg.get("register_alias")
    source = f"runs:/{run_id}/{source_artifact_path}"

    client = _mlflow_client()
    try:
        client.create_registered_model(model_name)
    except Exception as e:
        # Model may already exist in concurrent environments.
        err = str(e).lower()
        if "resource_already_exists" not in err and "already exists" not in err:
            raise
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


def set_registered_model_alias(*, model_name: str, alias: str, version: str) -> None:
    """Point a model alias to a specific registered version."""
    client = _mlflow_client()
    client.set_registered_model_alias(
        name=model_name,
        alias=str(alias),
        version=str(version),
    )


def _registry_config(config: dict[str, Any]) -> dict[str, Any]:
    mlflow_cfg = config.get("mlflow", {})
    if not isinstance(mlflow_cfg, dict):
        return {}
    registry_cfg = mlflow_cfg.get("registry", {})
    return registry_cfg if isinstance(registry_cfg, dict) else {}


def _resolve_registry_model_name(
    *, config: dict[str, Any], registry_cfg: dict[str, Any]
) -> str:
    template = str(
        registry_cfg.get("model_name_template")
        or registry_cfg.get("model_name")
        or "recsys-{model_name}-data-{data_version_short}"
    )
    model_cfg = config.get("model", {})
    model_name = (
        str(model_cfg.get("name"))
        if isinstance(model_cfg, dict) and model_cfg.get("name")
        else (
            str(model_cfg.get("variant"))
            if isinstance(model_cfg, dict) and model_cfg.get("variant")
            else (
                str(model_cfg.get("type"))
                if isinstance(model_cfg, dict) and model_cfg.get("type")
                else "srgnn"
            )
        )
    )
    data_version = _resolve_data_version(config)
    data_version_short = _short_data_version(data_version)
    return template.format(
        model_name=model_name,
        data_version=data_version,
        data_version_short=data_version_short,
    )


def _resolve_data_version(config: dict[str, Any]) -> str:
    lineage_cfg = config.get("lineage", {})
    if isinstance(lineage_cfg, dict):
        version = lineage_cfg.get("data_version")
        if version:
            return str(version)

    data_cfg = config.get("data", {})
    if isinstance(data_cfg, dict):
        for key in ("processed_path", "train_examples_path", "test_examples_path"):
            candidate = data_cfg.get(key)
            if not candidate:
                continue
            match = re.search(r"(v\d+[\w-]*)", str(candidate))
            if match:
                return match.group(1)
    return "default"


def _short_data_version(data_version: str) -> str:
    match = re.match(r"^(v\d+)", data_version)
    if match:
        return match.group(1)
    return data_version


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _mlflow_client():
    from mlflow.tracking import MlflowClient

    return MlflowClient()
