"""FastAPI application exposing recommendation endpoints."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RecSys API", version="0.1.0")


class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10


class RecommendResponse(BaseModel):
    user_id: int
    item_ids: list[int]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    """Return top-k recommendations for a user."""
    # TODO: load predictor and call get_recommendations
    raise NotImplementedError
