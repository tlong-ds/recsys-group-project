"""Recommendation orchestration: inference + catalog enrichment."""

from __future__ import annotations

import time

from loguru import logger
from prometheus_client import Counter, Histogram

from recsys.serving.catalog_repository import CatalogRepository
from recsys.serving.model_provider import (
    RECSYS_MODEL_LOAD_FAILURES_TOTAL,
    RECSYS_MODEL_READY,
    ModelProvider,
)
from recsys.serving.schemas import ProductInfo

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


class RecommendResult:
    """Internal value object returned by :class:`RecommendationService`."""

    __slots__ = ("recommendations", "recommended_products")

    def __init__(
        self,
        recommendations: list[int],
        recommended_products: list[ProductInfo],
    ) -> None:
        self.recommendations = recommendations
        self.recommended_products = recommended_products


class RecommendationService:
    """Orchestrate model inference and catalog metadata enrichment."""

    def __init__(
        self,
        model_provider: ModelProvider,
        catalog: CatalogRepository,
    ) -> None:
        self._model_provider = model_provider
        self._catalog = catalog

    async def recommend(
        self, item_sequence: list[int], top_k: int = 10
    ) -> RecommendResult:
        """Run full recommendation pipeline and record Prometheus metrics.

        Raises ``FileNotFoundError`` when the model is unavailable and generic
        ``Exception`` on unexpected inference errors — callers translate these
        into the appropriate HTTP status codes.
        """
        latency_start: float | None = None
        try:
            predictor, _ = self._model_provider.get_bundle()
            RECSYS_MODEL_READY.set(1)

            quality = predictor.input_quality(item_sequence)
            sequence_length = int(quality["sequence_length"])
            unknown_items = int(quality["unknown_items"])
            RECSYS_INPUT_SEQUENCE_LENGTH.observe(sequence_length)
            RECSYS_REQUESTED_TOP_K.observe(top_k)
            RECSYS_INPUT_ITEMS_TOTAL.inc(sequence_length)
            RECSYS_OOV_ITEMS_TOTAL.inc(unknown_items)

            latency_start = time.perf_counter()
            recommendations = predictor.get_recommendations(
                item_sequence, top_k=top_k
            )

            products = await self._enrich_with_metadata(recommendations)

            RECSYS_RECOMMENDATIONS_TOTAL.inc()
            RECSYS_RECOMMENDATION_REQUESTS_TOTAL.labels(status="success").inc()

            return RecommendResult(
                recommendations=recommendations,
                recommended_products=products,
            )
        except FileNotFoundError:
            RECSYS_MODEL_READY.set(0)
            RECSYS_MODEL_LOAD_FAILURES_TOTAL.inc()
            RECSYS_RECOMMENDATION_REQUESTS_TOTAL.labels(
                status="model_unavailable"
            ).inc()
            raise
        except Exception:
            RECSYS_RECOMMENDATION_REQUESTS_TOTAL.labels(status="error").inc()
            raise
        finally:
            if latency_start is not None:
                RECSYS_PREDICTION_LATENCY_SECONDS.observe(
                    time.perf_counter() - latency_start
                )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _enrich_with_metadata(
        self, recommendations: list[int]
    ) -> list[ProductInfo]:
        """Fetch catalog metadata for recommendation IDs, falling back on error."""
        if not recommendations:
            return []
        try:
            metadata_map = await self._catalog.fetch_product_metadata(recommendations)
            products: list[ProductInfo] = []
            for r_id in recommendations:
                if r_id in metadata_map:
                    products.append(metadata_map[r_id])
                else:
                    products.append(
                        ProductInfo(
                            id=r_id, categoryId=0, name=f"Product {r_id}", price=0.0
                        )
                    )
            return products
        except Exception as db_err:
            logger.warning(
                "Falling back to recommendation IDs after catalog lookup "
                "failure: {}",
                db_err,
            )
            return [
                ProductInfo(id=r_id, categoryId=0, name=f"Product {r_id}", price=0.0)
                for r_id in recommendations
            ]
