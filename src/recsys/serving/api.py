"""FastAPI application for session-based recommendations."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

from recsys.serving.catalog_repository import CatalogRepository, CatalogUnavailableError
from recsys.serving.evaluations import load_evaluation_metrics
from recsys.serving.event_sink import EventSink
from recsys.serving.model_provider import ModelProvider
from recsys.serving.recommendation_service import RecommendationService
from recsys.serving.schemas import (
    EvaluationsResponse,
    PaginatedProductsResponse,
    RecommendRequest,
    RecommendResponse,
    ViewLog,
)
from recsys.serving.security import (
    BodySizeLimitMiddleware,
    SecuritySettings,
    auth_dependency,
)
from recsys.utils.config import load_config

load_dotenv()

DEFAULT_CONFIG_PATH = Path("configs/serving_config.yaml")
CONFIG_ENV_VAR = "RECSYS_SERVING_CONFIG"
DEFAULT_CORS_ALLOWED_ORIGINS = (
    "http://0.0.0.0:5173",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


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

    # ---- Service wiring ----
    model_provider = ModelProvider(
        serving_config=config_serving,
        mlflow_config=config_mlflow,
        model_path=model_path,
    )
    catalog = CatalogRepository(
        db_url=os.getenv("NEON_DB_URL", "").strip() or None,
    )
    recommendation_service = RecommendationService(
        model_provider=model_provider,
        catalog=catalog,
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_allowed_origins(config_serving),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Instrument with Prometheus
    Instrumentator().instrument(app)

    # ---- Lifecycle ----

    @app.on_event("startup")
    async def startup_event() -> None:
        app.state.event_sink = None
        await catalog.connect()
        if catalog.available:
            event_sink = EventSink(catalog.pool)
            await event_sink.start()
            app.state.event_sink = event_sink

        if preload_on_startup:
            model_provider.preload()

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        await catalog.close()

    # ---- Routes ----

    @app.get("/metrics", include_in_schema=False)
    def metrics(_: str | None = Depends(verify_api_key)) -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health() -> dict[str, str]:
        return model_provider.health_payload()

    @app.get("/ready")
    def ready() -> dict[str, str]:
        try:
            return model_provider.readiness_payload()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Model unavailable.",
            ) from exc

    @app.get("/evaluations", response_model=EvaluationsResponse)
    def get_evaluations(
        _: str | None = Depends(verify_api_key),
    ) -> EvaluationsResponse:
        results = load_evaluation_metrics(Path("models/experiments"))
        return EvaluationsResponse(metrics=results)

    @app.get("/products", response_model=PaginatedProductsResponse)
    async def get_products(
        page: int = 1,
        page_size: int = 20,
        category_id: int | None = None,
        _: str | None = Depends(verify_api_key),
    ) -> PaginatedProductsResponse:
        """Fetch the product catalog from the database with offset-based pagination."""
        try:
            return await catalog.list_products(
                page=page, page_size=page_size, category_id=category_id
            )
        except CatalogUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/views", status_code=202)
    async def log_view(
        view: ViewLog,
        _: str | None = Depends(verify_api_key),
    ) -> dict[str, str]:
        """Asynchronously log a user item view to the batch queue."""
        event_sink: EventSink | None = getattr(app.state, "event_sink", None)
        if event_sink is None:
            from recsys.serving.event_sink import RECSYS_VIEW_LOG_FAILURES_TOTAL

            RECSYS_VIEW_LOG_FAILURES_TOTAL.inc()
            raise HTTPException(status_code=503, detail="Catalog database unavailable.")
        try:
            event_sink.enqueue(view)
            return {"status": "accepted"}
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=503, detail="Server is too busy to process view logs."
            )

    @app.post("/recommend", response_model=RecommendResponse)
    async def recommend(
        request: RecommendRequest,
        _: str | None = Depends(verify_api_key),
    ) -> RecommendResponse:
        try:
            result = await recommendation_service.recommend(
                request.item_sequence, request.top_k
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail="Recommendation failed."
            ) from exc
        return RecommendResponse(
            session_id=request.session_id,
            item_sequence=request.item_sequence,
            recommendations=result.recommendations,
            recommended_products=result.recommended_products,
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cors_allowed_origins(serving_config: dict[str, Any]) -> list[str]:
    raw_cors = serving_config.get("cors", {})
    cors_config = raw_cors if isinstance(raw_cors, dict) else {}
    raw_origins = cors_config.get("allowed_origins", DEFAULT_CORS_ALLOWED_ORIGINS)
    if isinstance(raw_origins, str):
        origins = [origin.strip().rstrip("/") for origin in raw_origins.split(",")]
    elif isinstance(raw_origins, list | tuple):
        origins = [str(origin).strip().rstrip("/") for origin in raw_origins]
    else:
        origins = [str(o).rstrip("/") for o in DEFAULT_CORS_ALLOWED_ORIGINS]

    allowed_origins = [origin for origin in origins if origin]
    return allowed_origins or list(DEFAULT_CORS_ALLOWED_ORIGINS)


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


def create_default_app() -> FastAPI:
    """Create an app from the configured default serving config."""
    config_path = Path(os.getenv(CONFIG_ENV_VAR, str(DEFAULT_CONFIG_PATH)))
    default_config = load_config(config_path) if config_path.exists() else {}
    return create_app(
        serving_config=default_config.get("serving", {}),
        mlflow_config=default_config.get("mlflow", {}),
    )
