"""Request and response schemas for the recommendation API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """Incoming request for next-item recommendation."""

    session_id: str | None = None
    item_sequence: list[int] = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class RecommendResponse(BaseModel):
    """Recommendation response payload."""

    session_id: str | None = None
    item_sequence: list[int]
    recommendations: list[int]
