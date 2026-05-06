"""Unit tests for EventSink — no FastAPI, no live DB."""

from __future__ import annotations

import asyncio

import pytest

from recsys.serving.event_sink import EventSink
from recsys.serving.schemas import ViewLog

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubConn:
    def __init__(self) -> None:
        self.executemany_calls: list[tuple] = []

    async def executemany(self, query: str, values: list) -> None:
        self.executemany_calls.append((query, values))


class _StubAcquire:
    def __init__(self, conn: _StubConn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *_args):
        return False


class _StubPool:
    def __init__(self) -> None:
        self.conn = _StubConn()

    def acquire(self):
        return _StubAcquire(self.conn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _view(session: str = "s1", item: int = 42) -> ViewLog:
    return ViewLog(sessionId=session, itemId=item)


def test_enqueue_accepts_view() -> None:
    sink = EventSink(_StubPool(), max_queue_size=10)

    sink.enqueue(_view())

    assert sink._queue.qsize() == 1


def test_enqueue_raises_when_full() -> None:
    sink = EventSink(_StubPool(), max_queue_size=1)
    sink.enqueue(_view())

    with pytest.raises(asyncio.QueueFull):
        sink.enqueue(_view(session="s2"))


def test_batch_writer_flushes_to_db() -> None:
    pool = _StubPool()
    sink = EventSink(pool, max_queue_size=100)

    async def _run() -> None:
        # Enqueue a view before starting the writer
        sink.enqueue(_view(session="batch-test", item=99))
        await sink.start()
        # Give the batch writer time to pick up and flush
        await asyncio.sleep(2.0)

    asyncio.get_event_loop().run_until_complete(_run())

    assert len(pool.conn.executemany_calls) >= 1
    _query, values = pool.conn.executemany_calls[0]
    assert values[0][0] == "batch-test"  # sessionId
    assert values[0][2] == 99            # itemId
