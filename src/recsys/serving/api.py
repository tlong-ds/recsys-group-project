"""FastAPI application for session-based recommendations."""

from __future__ import annotations

import argparse
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_fastapi_instrumentator import Instrumentator

from recsys.serving.predictor import Predictor
from recsys.serving.schemas import RecommendRequest, RecommendResponse
from recsys.serving.security import (
    BodySizeLimitMiddleware,
    SecuritySettings,
    auth_dependency,
)
from recsys.utils.config import load_config

DEFAULT_CONFIG_PATH = Path("configs/serving_config.yaml")
CONFIG_ENV_VAR = "RECSYS_SERVING_CONFIG"

# Custom metrics
RECSYS_RECOMMENDATIONS_TOTAL = Counter(
    "recsys_recommendations_total", "Total number of recommendations served"
)
RECSYS_RECOMMENDATION_REQUESTS_TOTAL = Counter(
    "recsys_recommendation_requests_total",
    "Recommendation requests by outcome status",
    ["status"],
)
RECSYS_OOV_ITEMS_TOTAL = Counter(
    "recsys_oov_items_total",
    "Total number of request items unknown to the loaded model catalog",
)
RECSYS_INPUT_ITEMS_TOTAL = Counter(
    "recsys_input_items_total",
    "Total number of input items received by the recommendation endpoint",
)
RECSYS_MODEL_LOAD_FAILURES_TOTAL = Counter(
    "recsys_model_load_failures_total",
    "Total number of failed model readiness/load checks",
)
RECSYS_PREDICTION_LATENCY_SECONDS = Histogram(
    "recsys_prediction_latency_seconds",
    "Latency of model recommendation calls",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
RECSYS_INPUT_SEQUENCE_LENGTH = Histogram(
    "recsys_input_sequence_length",
    "Length of item_sequence values received by the recommendation endpoint",
    buckets=(1, 2, 3, 5, 10, 20, 50, 100),
)
RECSYS_REQUESTED_TOP_K = Histogram(
    "recsys_requested_top_k",
    "Requested top_k values received by the recommendation endpoint",
    buckets=(1, 5, 10, 20, 50, 100),
)
RECSYS_MODEL_READY = Gauge(
    "recsys_model_ready",
    "Whether the serving process can load the configured model",
)
RECSYS_MODEL_READY.set(0)


def create_app(
    model_path: str | Path | None = None,
    serving_config: dict[str, Any] | None = None,
    mlflow_config: dict[str, Any] | None = None,
) -> FastAPI:
    """Create a FastAPI app bound to a specific model artifact."""
    config_serving = serving_config or {}
    config_mlflow = mlflow_config or {}
    preload_on_startup = bool(config_serving.get("preload_model_on_startup", True))
    security_settings = SecuritySettings.from_serving_config(config_serving)
    verify_api_key = auth_dependency(security_settings)
    resolved_model_path = str(
        model_path or config_serving.get("model_path") or _resolve_model_path()
    )
    app = FastAPI(
        title="RecSys API",
        version="0.1.0",
        docs_url="/docs" if security_settings.docs_enabled else None,
        redoc_url="/redoc" if security_settings.docs_enabled else None,
        openapi_url="/openapi.json" if security_settings.docs_enabled else None,
    )
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_body_bytes=security_settings.max_body_bytes,
    )

    # Instrument with Prometheus
    Instrumentator().instrument(app)

    @app.get("/metrics", include_in_schema=False)
    def metrics(_: str | None = Depends(verify_api_key)) -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @lru_cache(maxsize=1)
    def get_predictor_bundle() -> tuple[Predictor, dict[str, str]]:
        registry_cfg = config_serving.get("model_registry", {})
        if isinstance(registry_cfg, dict) and bool(registry_cfg.get("enabled", False)):
            model_name = str(registry_cfg.get("model_name", "recsys-srgnn"))
            model_alias = registry_cfg.get("model_alias")
            model_version = registry_cfg.get("model_version")
            artifact_path = str(registry_cfg.get("artifact_path", "registered_model"))
            cache_dir = registry_cfg.get("local_cache_dir")
            try:
                return Predictor.from_model_registry(
                    mlflow_config=config_mlflow,
                    model_name=model_name,
                    model_alias=str(model_alias) if model_alias else None,
                    model_version=str(model_version) if model_version else None,
                    artifact_path=artifact_path,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
            except Exception:
                if not bool(registry_cfg.get("fallback_to_filesystem", True)):
                    raise

        predictor = Predictor.from_path(resolved_model_path)
        return predictor, {
            "source": "filesystem",
            "artifact_path": resolved_model_path,
            "model_name": "",
            "model_version": "",
            "model_alias": "",
            "run_id": "",
        }

    @app.on_event("startup")
    def preload_model() -> None:
        if not preload_on_startup:
            return
        try:
            get_predictor_bundle()
            app.state.model_preload_error = ""
            RECSYS_MODEL_READY.set(1)
        except Exception as exc:
            # Keep the app bootable; /health will report degraded if load still fails.
            app.state.model_preload_error = str(exc)
            RECSYS_MODEL_READY.set(0)
            RECSYS_MODEL_LOAD_FAILURES_TOTAL.inc()

    @app.get("/health")
    def health() -> dict[str, str]:
        try:
            _, meta = get_predictor_bundle()
            RECSYS_MODEL_READY.set(1)
            return _model_status_payload("ok", meta)
        except Exception:
            RECSYS_MODEL_READY.set(0)
            return {
                "status": "degraded",
                "model_source": "unavailable",
            }

    @app.get("/ready")
    def ready() -> dict[str, str]:
        try:
            _, meta = get_predictor_bundle()
            RECSYS_MODEL_READY.set(1)
            return _model_status_payload("ready", meta)
        except Exception as exc:
            RECSYS_MODEL_READY.set(0)
            RECSYS_MODEL_LOAD_FAILURES_TOTAL.inc()
            raise HTTPException(
                status_code=503,
                detail="Model unavailable.",
            ) from exc

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(
        request: RecommendRequest,
        _: str | None = Depends(verify_api_key),
    ) -> RecommendResponse:
        latency_start: float | None = None
        try:
            predictor, _ = get_predictor_bundle()
            RECSYS_MODEL_READY.set(1)
            quality = predictor.input_quality(request.item_sequence)
            sequence_length = int(quality["sequence_length"])
            unknown_items = int(quality["unknown_items"])
            RECSYS_INPUT_SEQUENCE_LENGTH.observe(sequence_length)
            RECSYS_REQUESTED_TOP_K.observe(request.top_k)
            RECSYS_INPUT_ITEMS_TOTAL.inc(sequence_length)
            RECSYS_OOV_ITEMS_TOTAL.inc(unknown_items)

            latency_start = time.perf_counter()
            recommendations = predictor.get_recommendations(
                request.item_sequence,
                top_k=request.top_k,
            )
            # Track business metrics
            RECSYS_RECOMMENDATIONS_TOTAL.inc()
            RECSYS_RECOMMENDATION_REQUESTS_TOTAL.labels(status="success").inc()
        except FileNotFoundError as exc:
            RECSYS_MODEL_READY.set(0)
            RECSYS_MODEL_LOAD_FAILURES_TOTAL.inc()
            RECSYS_RECOMMENDATION_REQUESTS_TOTAL.labels(
                status="model_unavailable"
            ).inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            RECSYS_RECOMMENDATION_REQUESTS_TOTAL.labels(status="error").inc()
            raise HTTPException(
                status_code=500,
                detail="Recommendation failed.",
            ) from exc
        finally:
            if latency_start is not None:
                RECSYS_PREDICTION_LATENCY_SECONDS.observe(
                    time.perf_counter() - latency_start
                )
        return RecommendResponse(
            session_id=request.session_id,
            item_sequence=request.item_sequence,
            recommendations=recommendations,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve session-based recommendations")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    serving_config = config.get("serving", {})
    mlflow_config = config.get("mlflow", {})
    host = args.host or serving_config.get("host", "0.0.0.0")  # nosec B104
    port = args.port or int(serving_config.get("port", 8000))
    model_path = args.model_path or serving_config.get("model_path")
    reload_flag = bool(args.reload or serving_config.get("reload", False))
    if model_path:
        os.environ["RECSYS_MODEL_PATH"] = str(model_path)
    os.environ[CONFIG_ENV_VAR] = str(args.config)

    app_target: str | FastAPI = (
        "recsys.serving.api:create_default_app"
        if reload_flag
        else create_app(
            model_path=model_path,
            serving_config=serving_config,
            mlflow_config=mlflow_config,
        )
    )
    uvicorn.run(
        app_target,
        host=host,
        port=port,
        reload=reload_flag,
        factory=reload_flag,
    )


def _resolve_model_path() -> str:
    if os.getenv("RECSYS_MODEL_PATH"):
        return os.environ["RECSYS_MODEL_PATH"]
    if DEFAULT_CONFIG_PATH.exists():
        config = load_config(DEFAULT_CONFIG_PATH)
        return config.get("serving", {}).get("model_path", "models/trained/latest/")
    return "models/trained/latest/"


def _model_status_payload(status: str, meta: dict[str, str]) -> dict[str, str]:
    return {
        "status": status,
        "model_source": meta.get("source", ""),
        "model_name": meta.get("model_name", ""),
        "model_version": meta.get("model_version", ""),
        "model_alias": meta.get("model_alias", ""),
        "cache_hit": meta.get("cache_hit", ""),
    }


def create_default_app() -> FastAPI:
    """Create an app from the configured default serving config."""
    config_path = Path(os.getenv(CONFIG_ENV_VAR, str(DEFAULT_CONFIG_PATH)))
    default_config = load_config(config_path) if config_path.exists() else {}
    return create_app(
        serving_config=default_config.get("serving", {}),
        mlflow_config=default_config.get("mlflow", {}),
    )
