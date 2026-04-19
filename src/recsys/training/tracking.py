"""Optional MLflow tracking utilities with DagsHub integration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow

from recsys.models.srgnn import SRGNNRecommender

DEFAULT_REPO_OWNER = "lytlong.pers"
DEFAULT_REPO_NAME = "recsys-group-project"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _mlflow_config(config: dict[str, Any]) -> dict[str, Any]:
    loaded = config.get("mlflow", {})
    return loaded if isinstance(loaded, dict) else {}


def _dagshub_config(config: dict[str, Any]) -> dict[str, Any]:
    loaded = _mlflow_config(config).get("dagshub", {})
    return loaded if isinstance(loaded, dict) else {}


def tracking_enabled(config: dict[str, Any]) -> bool:
    return _as_bool(_mlflow_config(config).get("enabled", False))


def dagshub_repo(config: dict[str, Any]) -> tuple[str, str]:
    dag_cfg = _dagshub_config(config)
    owner = str(dag_cfg.get("repo_owner", DEFAULT_REPO_OWNER))
    name = str(dag_cfg.get("repo_name", DEFAULT_REPO_NAME))
    return owner, name


def configure_tracking(config: dict[str, Any]) -> None:
    if not tracking_enabled(config):
        return

    mlflow_cfg = _mlflow_config(config)
    dag_cfg = _dagshub_config(config)
    if _as_bool(dag_cfg.get("enabled", False)):
        owner, name = dagshub_repo(config)
        if not owner or not name:
            raise ValueError(
                "DagsHub tracking is enabled, but mlflow.dagshub.repo_owner/repo_name are missing."
            )
        try:
            import dagshub
        except ImportError as exc:
            raise RuntimeError(
                "DagsHub tracking is enabled, but the "
                "'dagshub' package is not installed."
            ) from exc
        _configure_mlflow_auth_for_dagshub(dag_cfg)
        dagshub.init(repo_owner=owner, repo_name=name, mlflow=True)
    else:
        tracking_uri = mlflow_cfg.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(str(tracking_uri))

    experiment_name = mlflow_cfg.get("experiment_name")
    if experiment_name:
        mlflow.set_experiment(str(experiment_name))


def configure_system_metrics(config: dict[str, Any]) -> None:
    """Configure global MLflow system metrics behavior from config."""
    mlflow_cfg = _mlflow_config(config)
    sys_cfg = mlflow_cfg.get("system_metrics", {})
    if not isinstance(sys_cfg, dict):
        return

    if "enabled" in sys_cfg:
        if _as_bool(sys_cfg.get("enabled")):
            mlflow.enable_system_metrics_logging()
        else:
            mlflow.disable_system_metrics_logging()
    if "sampling_interval" in sys_cfg and sys_cfg.get("sampling_interval") is not None:
        mlflow.set_system_metrics_sampling_interval(int(sys_cfg["sampling_interval"]))
    if "samples_before_logging" in sys_cfg and sys_cfg.get("samples_before_logging") is not None:
        mlflow.set_system_metrics_samples_before_logging(int(sys_cfg["samples_before_logging"]))


def system_metrics_run_override(config: dict[str, Any]) -> bool | None:
    """Return per-run log_system_metrics override when configured."""
    mlflow_cfg = _mlflow_config(config)
    sys_cfg = mlflow_cfg.get("system_metrics", {})
    if not isinstance(sys_cfg, dict):
        return None
    if "enabled" not in sys_cfg:
        return None
    return _as_bool(sys_cfg.get("enabled"))


def _sanitize_metric_key(name: str) -> str:
    return name.replace("@", "_at_").replace("/", "_")


def sanitize_metric_key(name: str) -> str:
    return _sanitize_metric_key(name)


def _configure_mlflow_auth_for_dagshub(dag_cfg: dict[str, Any]) -> None:
    """Bridge token env vars for both DagsHub SDK and MLflow auth."""
    token_env_name = str(dag_cfg.get("token_env_var", "DAGSHUB_USER_TOKEN"))
    username_env_name = str(dag_cfg.get("username_env_var", "DAGSHUB_USERNAME"))
    password_env_name = str(dag_cfg.get("password_env_var", "DAGSHUB_USER_TOKEN"))

    username = os.getenv(username_env_name)
    token = os.getenv(token_env_name) or os.getenv("DAGSHUB_USER_TOKEN")
    password = os.getenv(password_env_name) or token

    # dagshub.init() resolves auth from DAGSHUB_USER_TOKEN; map legacy token env if needed.
    if token and not os.getenv("DAGSHUB_USER_TOKEN"):
        os.environ["DAGSHUB_USER_TOKEN"] = token

    if username and not os.getenv("MLFLOW_TRACKING_USERNAME"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
    if password and not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password


def _training_params(config: dict[str, Any], model: SRGNNRecommender) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    params: dict[str, Any] = {
        "model_name": model.model_name,
        "model_version": model.model_version,
        "embedding_dim": int(model_cfg.get("embedding_dim", model.embedding_dim)),
        "hidden_size": int(model_cfg.get("hidden_size", model.hidden_size)),
        "max_session_length": int(
            model_cfg.get("max_session_length", model.max_session_length)
        ),
        "fallback_weight": float(
            model_cfg.get("fallback_weight", model.fallback_weight)
        ),
        "top_k": int(training_cfg.get("top_k", 20)),
    }
    return params


def log_training_run(
    *,
    config: dict[str, Any],
    model: SRGNNRecommender,
    metrics: dict[str, float],
    artifact_path: Path,
) -> None:
    if not tracking_enabled(config):
        return

    configure_tracking(config)
    run_name = f"train-{model.model_name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_training_params(config, model))
        if metrics:
            mlflow.log_metrics(
                {f"val_{_sanitize_metric_key(k)}": float(v) for k, v in metrics.items()}
            )
        mlflow.log_param("artifact_path", str(artifact_path))
        artifact_dir = (
            artifact_path.parent if artifact_path.is_file() else artifact_path
        )
        mlflow.log_artifacts(str(artifact_dir), artifact_path="model")


def log_evaluation_run(
    *,
    config: dict[str, Any],
    metrics: dict[str, float],
    model_path: Path,
) -> None:
    if not tracking_enabled(config):
        return

    configure_tracking(config)
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics(
            {f"test_{_sanitize_metric_key(k)}": float(v) for k, v in metrics.items()}
        )
        mlflow.log_param("model_path", str(model_path))
