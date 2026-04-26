"""Product catalog database access."""

from __future__ import annotations

from typing import Any

import asyncpg
from loguru import logger
from prometheus_client import Counter

from recsys.serving.schemas import PaginatedProductsResponse, ProductInfo

RECSYS_CATALOG_LOOKUP_FAILURES_TOTAL = Counter(
    "recsys_catalog_lookup_failures_total",
    "Total number of failed catalog metadata lookups",
)


class CatalogUnavailableError(Exception):
    """Raised when the catalog database pool is not available."""


class CatalogRepository:
    """Async wrapper around the product catalog in Neon/PostgreSQL."""

    def __init__(self, db_url: str | None = None) -> None:
        self._db_url = db_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create the connection pool.  Logs a warning on failure."""
        if not self._db_url:
            logger.warning("NEON_DB_URL is not configured; catalog DB access disabled")
            return
        try:
            self._pool = await asyncpg.create_pool(
                self._db_url, min_size=2, max_size=10
            )
        except Exception as exc:
            logger.warning("Catalog database pool initialization failed: {}", exc)

    async def close(self) -> None:
        """Drain and close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> Any:
        """Raw pool reference (for sharing with ``EventSink``)."""
        return self._pool

    @property
    def available(self) -> bool:
        return self._pool is not None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def fetch_product_metadata(
        self, item_ids: list[int]
    ) -> dict[int, ProductInfo]:
        """Return a ``{item_id: ProductInfo}`` map for the given IDs.

        Raises on pool unavailability or query errors, incrementing the
        catalog-lookup-failure counter in both cases.
        """
        if not self.available:
            RECSYS_CATALOG_LOOKUP_FAILURES_TOTAL.inc()
            raise CatalogUnavailableError("Catalog database unavailable")
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                rows = await conn.fetch(
                    'SELECT p."itemId" as "id", pc."categoryId" '
                    'as "categoryId", '
                    'p.product_name_tokens as "name", '
                    '(POWER(2, p.pricelog2) - 1) as "price" '
                    "FROM products p "
                    'JOIN product_categories pc ON p."itemId" = pc."itemId" '
                    'WHERE p."itemId" = ANY($1)',
                    item_ids,
                )
                return {row["id"]: ProductInfo(**dict(row)) for row in rows}
        except Exception:
            RECSYS_CATALOG_LOOKUP_FAILURES_TOTAL.inc()
            raise

    async def list_products(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        category_id: int | None = None,
    ) -> PaginatedProductsResponse:
        """Return a paginated product listing.

        Raises ``CatalogUnavailableError`` when the pool is ``None``.
        """
        if not self.available:
            raise CatalogUnavailableError("Catalog database unavailable")

        async with self._pool.acquire() as conn:  # type: ignore[union-attr]
            where_clauses: list[str] = []
            params: list[Any] = []

            if category_id is not None:
                where_clauses.append(f'pc."categoryId" = ${len(params) + 1}')
                params.append(category_id)

            where_sql = (
                f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            )

            count_query = (
                f"SELECT COUNT(*) FROM products p "
                f'JOIN product_categories pc ON p."itemId" = pc."itemId" '
                f"{where_sql}"  # nosec B608
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
                f"{where_sql} "  # nosec B608
                'ORDER BY p."itemId" ASC '
                f"LIMIT {limit_param} OFFSET {offset_param}"  # nosec B608
            )

            rows = await conn.fetch(query, *params)
            items = [ProductInfo(**dict(row)) for row in rows]

            next_page = page + 1 if offset + len(items) < total_count else None
            total_pages = (total_count + page_size - 1) // page_size

            return PaginatedProductsResponse(
                items=items,
                total_pages=total_pages,
                current_page=page,
                next_cursor=next_page,
            )
