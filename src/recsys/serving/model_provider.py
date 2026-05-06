"""Model loading, registry resolution, and readiness management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger
from prometheus_client import Counter, Gauge

from recsys.serving.predictor import Predictor
from recsys.utils.config import load_config

DEFAULT_CONFIG_PATH = Path("configs/serving_config.yaml")

RECSYS_MODEL_READY = Gauge(
    "recsys_model_ready",
    "Whether the serving process can load the configured model",
)
RECSYS_MODEL_READY.set(0)

RECSYS_MODEL_LOAD_FAILURES_TOTAL = Counter(
    "recsys_model_load_failures_total",
    "Total number of failed model readiness/load checks",
)


class ModelProvider:
    """Load, cache, and expose a serving model artifact."""

    def __init__(
        self,
        *,
        serving_config: dict[str, Any],
        mlflow_config: dict[str, Any],
        model_path: str | Path | None = None,
    ) -> None:
        self._serving_config = serving_config
        self._mlflow_config = mlflow_config
        self._model_path = model_path
        self._bundle: tuple[Predictor, dict[str, str]] | None = None

    def get_bundle(self) -> tuple[Predictor, dict[str, str]]:
        """Return the cached ``(Predictor, metadata)`` pair, loading on first call."""
        if self._bundle is not None:
            return self._bundle

        registry_cfg = self._serving_config.get("model_registry", {})
        if isinstance(registry_cfg, dict) and bool(registry_cfg.get("enabled", False)):
            deploy_overrides = _deploy_registry_overrides()
            model_name = str(
                deploy_overrides.get("model_name")
                or registry_cfg.get("model_name", "recsys-serving")
            )
            model_alias = (
                registry_cfg.get("model_alias")
                if not deploy_overrides.get("model_version")
                else None
            )
            model_version = deploy_overrides.get("model_version") or registry_cfg.get(
                "model_version"
            )
            run_id = deploy_overrides.get("run_id")
            artifact_path = str(registry_cfg.get("artifact_path", "registered_model"))
            cache_dir = (
                os.getenv("RECSYS_MODEL_CACHE_ROOT")
                or deploy_overrides.get("cache_root")
                or registry_cfg.get("local_cache_dir")
            )
            overrides = deploy_overrides
            has_pins = bool(
                overrides.get("model_name")
                or overrides.get("model_version")
                or overrides.get("run_id")
            )
            if not has_pins and not bool(
                registry_cfg.get("fallback_to_filesystem", True)
            ):
                logger.warning(
                    "Registry-first mode with fallback_to_filesystem=false, "
                    "but no RECSYS_DEPLOY_MODEL_NAME/VERSION/RUN_ID env vars are set. "
                    "The model will resolve via the configured alias '{}' only.",
                    model_alias,
                )

            try:
                bundle = Predictor.from_model_registry(
                    mlflow_config=self._mlflow_config,
                    model_name=model_name,
                    model_alias=str(model_alias) if model_alias else None,
                    model_version=str(model_version) if model_version else None,
                    run_id=str(run_id) if run_id else None,
                    artifact_path=artifact_path,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
                self._bundle = bundle
                logger.info(
                    "Model loaded from registry: name={}, version={}, run_id={}",
                    bundle[1].get("model_name", ""),
                    bundle[1].get("model_version", ""),
                    bundle[1].get("run_id", ""),
                )
                return bundle
            except Exception:
                if not bool(registry_cfg.get("fallback_to_filesystem", True)):
                    raise

        resolved = _resolve_model_path(
            explicit_model_path=self._model_path,
            serving_config=self._serving_config,
        )
        predictor = Predictor.from_path(resolved)
        bundle = predictor, {
            "source": "filesystem",
            "artifact_path": resolved,
            "model_name": "",
            "model_version": "",
            "run_id": "",
        }
        self._bundle = bundle
        return bundle

    def preload(self) -> None:
        """Eagerly load the model so ``/health`` responds immediately."""
        try:
            self.get_bundle()
            RECSYS_MODEL_READY.set(1)
        except Exception as exc:
            RECSYS_MODEL_READY.set(0)
            RECSYS_MODEL_LOAD_FAILURES_TOTAL.inc()
            logger.warning("Model preload failed: {}", exc)

    def model_identity(self) -> dict[str, str] | None:
        """Return the loaded model's identity metadata, or *None* if not yet loaded."""
        if self._bundle is None:
            return None
        return dict(self._bundle[1])

    def health_payload(self) -> dict[str, str]:
        """Return a health-check dict.  Never raises."""
        try:
            _, meta = self.get_bundle()
            RECSYS_MODEL_READY.set(1)
            return _model_status_payload("ok", meta)
        except Exception:
            RECSYS_MODEL_READY.set(0)
            return {"status": "degraded", "model_source": "unavailable"}

    def readiness_payload(self) -> dict[str, str]:
        """Return a readiness-check dict.  Raises if the model cannot load."""
        try:
            _, meta = self.get_bundle()
            RECSYS_MODEL_READY.set(1)
            return _model_status_payload("ready", meta)
        except Exception:
            RECSYS_MODEL_READY.set(0)
            RECSYS_MODEL_LOAD_FAILURES_TOTAL.inc()
            raise


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _deploy_registry_overrides() -> dict[str, str]:
    return {
        "model_name": os.getenv("RECSYS_DEPLOY_MODEL_NAME", "").strip(),
        "model_version": os.getenv("RECSYS_DEPLOY_MODEL_VERSION", "").strip(),
        "run_id": os.getenv("RECSYS_DEPLOY_RUN_ID", "").strip(),
        "cache_root": os.getenv("RECSYS_MODEL_CACHE_ROOT", "").strip(),
    }


def _resolve_model_path(
    *,
    explicit_model_path: str | Path | None = None,
    serving_config: dict[str, Any] | None = None,
) -> str:
    env_path = os.getenv("RECSYS_MODEL_PATH", "").strip()
    if env_path:
        return env_path
    if explicit_model_path:
        return str(explicit_model_path)
    cfg_path = (
        str((serving_config or {}).get("model_path", "")).strip()
        if isinstance(serving_config, dict)
        else ""
    )
    if cfg_path:
        return cfg_path
    if DEFAULT_CONFIG_PATH.exists():
        config = load_config(DEFAULT_CONFIG_PATH)
        discovered = str(config.get("serving", {}).get("model_path", "")).strip()
        if discovered:
            return discovered
    raise ValueError(
        "Filesystem model loading requested, but no serving.model_path was configured."
    )


def _model_status_payload(status: str, meta: dict[str, str]) -> dict[str, str]:
    return {
        "status": status,
        "model_source": meta.get("source", ""),
        "model_name": meta.get("model_name", ""),
        "model_version": meta.get("model_version", ""),
        "run_id": meta.get("run_id", ""),
    }
