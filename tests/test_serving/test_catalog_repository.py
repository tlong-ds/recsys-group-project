"""Unit tests for CatalogRepository — no FastAPI, no live DB."""

from __future__ import annotations

import asyncio

import pytest

from recsys.serving.catalog_repository import CatalogRepository, CatalogUnavailableError
from recsys.serving.schemas import ProductInfo


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubConn:
    """Simulates an asyncpg connection for fetch/fetchval."""

    def __init__(self, *, rows: list[dict] | None = None, fail: bool = False) -> None:
        self._rows = rows or []
        self._fail = fail

    async def fetch(self, *_args, **_kwargs):
        if self._fail:
            raise RuntimeError("query failed")
        return self._rows

    async def fetchval(self, *_args, **_kwargs):
        if self._fail:
            raise RuntimeError("query failed")
        return len(self._rows)


class _StubAcquire:
    def __init__(self, conn: _StubConn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *_args):
        return False


class _StubPool:
    def __init__(self, conn: _StubConn) -> None:
        self._conn = conn

    def acquire(self):
        return _StubAcquire(self._conn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_available_false_when_no_pool() -> None:
    repo = CatalogRepository(db_url=None)
    assert repo.available is False


def test_fetch_product_metadata_raises_when_unavailable() -> None:
    repo = CatalogRepository(db_url=None)

    with pytest.raises(CatalogUnavailableError):
        asyncio.get_event_loop().run_until_complete(
            repo.fetch_product_metadata([1, 2])
        )


def test_fetch_product_metadata_returns_map() -> None:
    rows = [
        {"id": 10, "categoryId": 5, "name": "Widget", "price": 12.5},
        {"id": 20, "categoryId": 3, "name": "Gadget", "price": 25.0},
    ]
    repo = CatalogRepository(db_url=None)
    repo._pool = _StubPool(_StubConn(rows=rows))

    result = asyncio.get_event_loop().run_until_complete(
        repo.fetch_product_metadata([10, 20])
    )

    assert len(result) == 2
    assert result[10] == ProductInfo(id=10, categoryId=5, name="Widget", price=12.5)
    assert result[20] == ProductInfo(id=20, categoryId=3, name="Gadget", price=25.0)


def test_fetch_product_metadata_increments_counter_on_failure() -> None:
    from recsys.serving.catalog_repository import RECSYS_CATALOG_LOOKUP_FAILURES_TOTAL

    repo = CatalogRepository(db_url=None)
    repo._pool = _StubPool(_StubConn(fail=True))

    before = RECSYS_CATALOG_LOOKUP_FAILURES_TOTAL._value.get()

    with pytest.raises(RuntimeError):
        asyncio.get_event_loop().run_until_complete(
            repo.fetch_product_metadata([1])
        )

    assert RECSYS_CATALOG_LOOKUP_FAILURES_TOTAL._value.get() > before


def test_list_products_raises_when_unavailable() -> None:
    repo = CatalogRepository(db_url=None)

    with pytest.raises(CatalogUnavailableError):
        asyncio.get_event_loop().run_until_complete(
            repo.list_products(page=1, page_size=10)
        )
