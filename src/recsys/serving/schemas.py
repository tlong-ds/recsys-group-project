"""Request and response schemas for the recommendation API."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

PositiveItemId = Annotated[int, Field(ge=1)]


class RecommendRequest(BaseModel):
    """Incoming request for next-item recommendation."""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(default=None, max_length=128)
    item_sequence: list[PositiveItemId] = Field(min_length=1, max_length=100)
    top_k: int = Field(default=10, ge=1, le=100)


class RecommendResponse(BaseModel):
    """Recommendation response payload."""

    session_id: str | None = None
    item_sequence: list[int]
    recommendations: list[int]
