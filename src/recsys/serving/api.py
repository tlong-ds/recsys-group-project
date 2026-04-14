"""FastAPI application for session-based recommendations."""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from recsys.serving.predictor import Predictor
from recsys.serving.schemas import RecommendRequest, RecommendResponse
from recsys.utils.config import load_config

DEFAULT_CONFIG_PATH = Path("configs/serving_config.yaml")

# Custom metrics
RECSYS_RECOMMENDATIONS_TOTAL = Counter(
    "recsys_recommendations_total", "Total number of recommendations served"
)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Create a FastAPI app bound to a specific model artifact."""
    resolved_model_path = str(model_path or _resolve_model_path())
    app = FastAPI(title="RecSys API", version="0.1.0")

    # Instrument with Prometheus
    Instrumentator().instrument(app).expose(app)

    @lru_cache(maxsize=1)
    def get_predictor() -> Predictor:
        return Predictor.from_path(resolved_model_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        status = "ok" if Path(resolved_model_path).exists() else "degraded"
        return {"status": status, "model_path": resolved_model_path}

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(request: RecommendRequest) -> RecommendResponse:
        try:
            recommendations = get_predictor().get_recommendations(
                request.item_sequence,
                top_k=request.top_k,
            )
            # Track business metrics
            RECSYS_RECOMMENDATIONS_TOTAL.inc()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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
    host = args.host or serving_config.get("host", "0.0.0.0")
    port = args.port or int(serving_config.get("port", 8000))
    model_path = args.model_path or serving_config.get("model_path")
    reload_flag = bool(args.reload or serving_config.get("reload", False))
    if model_path:
        os.environ["RECSYS_MODEL_PATH"] = str(model_path)

    app_target: str | FastAPI = "recsys.serving.api:app" if reload_flag else create_app(model_path)
    uvicorn.run(app_target, host=host, port=port, reload=reload_flag)


def _resolve_model_path() -> str:
    if os.getenv("RECSYS_MODEL_PATH"):
        return os.environ["RECSYS_MODEL_PATH"]
    if DEFAULT_CONFIG_PATH.exists():
        config = load_config(DEFAULT_CONFIG_PATH)
        return config.get("serving", {}).get("model_path", "models/trained/latest/")
    return "models/trained/latest/"


app = create_app()
