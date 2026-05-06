"""Unit tests for RecommendationService — no FastAPI, no live DB."""

from __future__ import annotations

import asyncio

import pytest

from recsys.serving.recommendation_service import RecommendationService
from recsys.serving.schemas import ProductInfo

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubPredictor:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self._item_to_idx = {1: 1, 2: 2, 3: 3}

    def get_recommendations(
        self, item_sequence: list[int], top_k: int = 10
    ) -> list[int]:
        if self._fail:
            raise RuntimeError("boom")
        return list(range(10, 10 + top_k))

    def input_quality(self, item_sequence: list[int]) -> dict[str, int | float]:
        unknown = sum(1 for i in item_sequence if i not in self._item_to_idx)
        seq_len = len(item_sequence)
        return {
            "sequence_length": seq_len,
            "known_items": seq_len - unknown,
            "unknown_items": unknown,
            "oov_ratio": unknown / seq_len if seq_len else 0.0,
            "known_catalog_items": len(self._item_to_idx),
        }


class _StubModelProvider:
    """Mimics ModelProvider.get_bundle() without touching the filesystem."""

    def __init__(self, predictor: _StubPredictor) -> None:
        self._predictor = predictor

    def get_bundle(self):
        return self._predictor, {"source": "stub"}


class _StubCatalog:
    """Mimics CatalogRepository.fetch_product_metadata()."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    async def fetch_product_metadata(
        self, item_ids: list[int]
    ) -> dict[int, ProductInfo]:
        if self._fail:
            raise RuntimeError("catalog down")
        return {
            item_id: ProductInfo(
                id=item_id, categoryId=1, name=f"Item {item_id}", price=9.99
            )
            for item_id in item_ids
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recommend_returns_products_with_metadata() -> None:
    svc = RecommendationService(
        model_provider=_StubModelProvider(_StubPredictor()),
        catalog=_StubCatalog(),
    )

    result = asyncio.get_event_loop().run_until_complete(svc.recommend([1, 2], top_k=3))

    assert result.recommendations == [10, 11, 12]
    assert len(result.recommended_products) == 3
    assert result.recommended_products[0].name == "Item 10"
    assert result.recommended_products[0].price == 9.99


def test_recommend_falls_back_on_catalog_failure() -> None:
    svc = RecommendationService(
        model_provider=_StubModelProvider(_StubPredictor()),
        catalog=_StubCatalog(fail=True),
    )

    result = asyncio.get_event_loop().run_until_complete(svc.recommend([1, 2], top_k=2))

    assert result.recommendations == [10, 11]
    # Fallback products use generic names
    assert result.recommended_products[0].name == "Product 10"
    assert result.recommended_products[0].price == 0.0


def test_recommend_raises_on_model_unavailable() -> None:
    class _FailingProvider:
        def get_bundle(self):
            raise FileNotFoundError("model missing")

    svc = RecommendationService(
        model_provider=_FailingProvider(),
        catalog=_StubCatalog(),
    )

    with pytest.raises(FileNotFoundError, match="model missing"):
        asyncio.get_event_loop().run_until_complete(svc.recommend([1], top_k=1))


def test_recommend_raises_on_inference_error() -> None:
    svc = RecommendationService(
        model_provider=_StubModelProvider(_StubPredictor(fail=True)),
        catalog=_StubCatalog(),
    )

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.get_event_loop().run_until_complete(svc.recommend([1], top_k=1))
