"""FastAPI application for session-based recommendations."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import asyncpg
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_fastapi_instrumentator import Instrumentator

from recsys.serving.predictor import Predictor
from recsys.serving.schemas import (
    EvaluationsResponse,
    ModelMetrics,
    PaginatedProductsResponse,
    ProductInfo,
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
DB_URL = os.getenv("NEON_DB_URL")


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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Instrument with Prometheus
    Instrumentator().instrument(app)

    @app.on_event("startup")
    async def startup_event() -> None:
        # Initialize DB pool for async operations
        app.state.pool = await asyncpg.create_pool(DB_URL, min_size=2, max_size=10)
        app.state.view_queue = asyncio.Queue(maxsize=10000)
        
        # Start background writer task for high-throughput views
        asyncio.create_task(batch_writer_task(app))
        
        if preload_on_startup:
            preload_model()

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        if hasattr(app.state, "pool"):
            await app.state.pool.close()

    async def batch_writer_task(app: FastAPI) -> None:
        """Background task to batch write user views to DB without blocking the API."""
        while True:
            batch = []
            try:
                # Wait for first item to arrive
                view = await app.state.view_queue.get()
                batch.append(view)
                
                # Collect more items for up to 1 second or until batch size 1000
                start_collect = time.time()
                while len(batch) < 1000 and (time.time() - start_collect) < 1.0:
                    try:
                        view = app.state.view_queue.get_nowait()
                        batch.append(view)
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.1)
                
                if batch:
                    async with app.state.pool.acquire() as conn:
                        query = (
                            "INSERT INTO user_views "
                            '("sessionId", "userId", "itemId", timeframe, eventdate) '
                            "VALUES ($1, $2, $3, $4, $5)"
                        )
                        values = [
                            (
                                v.sessionId, 
                                v.userId, 
                                v.itemId, 
                                v.timeframe or int(time.time() * 1000),
                                v.eventdate or time.strftime("%Y-%m-%d")
                            )
                            for v in batch
                        ]
                        await conn.executemany(query, values)
            except Exception as e:
                # In production, use structured logging here
                print(f"Error in batch_writer_task: {e}")
            
            # Yield control to prevent CPU spinning if queue was empty
            await asyncio.sleep(0.5)

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

    @app.get("/evaluations", response_model=EvaluationsResponse)
    def get_evaluations() -> EvaluationsResponse:
        import json
        
        experiments_dir = Path("models/experiments")
        results: list[ModelMetrics] = []
        if not experiments_dir.exists():
            return EvaluationsResponse(metrics=results)
        
        for data_version_dir in experiments_dir.iterdir():
            if not data_version_dir.is_dir():
                continue
            for model_dir in data_version_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                metrics_file = model_dir / "latest" / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file) as f:
                            metrics = json.load(f)
                        
                        profile = model_dir.name.upper()
                        if profile.startswith("SRGNN_"):
                            profile = f"SR-GNN ({profile.split('_')[1].upper()})"
                        elif profile == "SRGNN":
                            profile = "SR-GNN"
                        elif profile == "TAGNN":
                            profile = "TAGNN"
                        elif profile == "GGNN":
                            profile = "GGNN"
                        
                        results.append(ModelMetrics(
                            profile=profile,
                            dataVersion=data_version_dir.name,
                            hrAtK=metrics.get("hr@k", 0.0),
                            mrrAtK=metrics.get("mrr@k", 0.0)
                        ))
                    except Exception:
                        pass
        
        return EvaluationsResponse(metrics=results)

    @app.get("/products", response_model=PaginatedProductsResponse)
    async def get_products(
        page: int = 1,
        page_size: int = 20,
        category_id: int | None = None
    ) -> PaginatedProductsResponse:
        """Fetch the product catalog from the database with offset-based pagination."""
        async with app.state.pool.acquire() as conn:
            where_clauses = []
            params = []
            
            if category_id is not None:
                where_clauses.append(f'pc."categoryId" = ${len(params) + 1}')
                params.append(category_id)
                
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            
            # Get total count for pagination metadata
            count_query = (
                f'SELECT COUNT(*) FROM products p '
                f'JOIN product_categories pc ON p."itemId" = pc."itemId" '
                f'{where_sql}'
            )
            total_count = await conn.fetchval(count_query, *params)
            
            offset = (page - 1) * page_size
            limit_param = f"${len(params) + 1}"
            params.append(page_size)
            offset_param = f"${len(params) + 1}"
            params.append(offset)
            
            query = (
                'SELECT p."itemId" as "id", pc."categoryId" as "categoryId", '
                'p.product_name_tokens as "name", '
                '(POWER(2, p.pricelog2) - 1) as "price" '
                "FROM products p "
                'JOIN product_categories pc ON p."itemId" = pc."itemId" '
                f"{where_sql} "
                'ORDER BY p."itemId" ASC '
                f"LIMIT {limit_param} OFFSET {offset_param}"
            )
            
            rows = await conn.fetch(query, *params)
            items = [ProductInfo(**dict(row)) for row in rows]
            
            # In offset-based, we return next_page instead of cursor
            next_page = page + 1 if offset + len(items) < total_count else None
            total_pages = (total_count + page_size - 1) // page_size
            
            return PaginatedProductsResponse(
                items=items, 
                total_pages=total_pages,
                current_page=page,
                next_cursor=next_page
            )

    @app.post("/views", status_code=202)
    async def log_view(view: ViewLog) -> dict[str, str]:
        """Asynchronously log a user item view to the batch queue."""
        try:
            app.state.view_queue.put_nowait(view)
            return {"status": "accepted"}
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=503, detail="Server is too busy to process view logs."
            )

    @app.post("/recommend", response_model=RecommendResponse)
    async def recommend(
        request: RecommendRequest,
        api_request: Request,
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
            
            # Fetch metadata for recommendations
            recommended_products = []
            if recommendations:
                try:
                    async with api_request.app.state.pool.acquire() as conn:
                        rows = await conn.fetch(
                            'SELECT p."itemId" as "id", pc."categoryId" '
                            'as "categoryId", '
                            'p.product_name_tokens as "name", '
                            '(POWER(2, p.pricelog2) - 1) as "price" '
                            "FROM products p "
                            'JOIN product_categories pc ON p."itemId" = pc."itemId" '
                            'WHERE p."itemId" = ANY($1)',
                            recommendations,
                        )
                        # Debug logging to investigate missing metadata
                        print(
                            f"DEBUG: Found {len(rows)} metadata rows for "
                            f"{len(recommendations)} recommendations"
                        )
                        if rows:
                            print(f"DEBUG: First row keys: {list(rows[0].keys())}")
                        
                        # Maintain order of recommendations
                        metadata_map = {row['id']: row for row in rows}
                        for r_id in recommendations:
                            if r_id in metadata_map:
                                row = metadata_map[r_id]
                                recommended_products.append(ProductInfo(
                                    id=row['id'],
                                    categoryId=row['categoryId'],
                                    name=row['name'],
                                    price=row['price']
                                ))
                            else:
                                # Fallback if metadata missing
                                recommended_products.append(ProductInfo(
                                    id=r_id,
                                    categoryId=0,
                                    name=f"Product {r_id}",
                                    price=0.0
                                ))
                except Exception as db_err:
                    print(f"Error fetching metadata for recommendations: {db_err}")
                    # Fallback to simple list if DB fails
                    recommended_products = [
                        ProductInfo(
                            id=r_id, categoryId=0, name=f"Product {r_id}", price=0.0
                        )
                        for r_id in recommendations
                    ]

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
            recommended_products=recommended_products
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
