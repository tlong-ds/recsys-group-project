"""Async event sink for user view logging."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger
from prometheus_client import Counter

from recsys.serving.schemas import ViewLog

RECSYS_VIEW_LOG_FAILURES_TOTAL = Counter(
    "recsys_view_log_failures_total",
    "Total number of failed user view log attempts",
)


class EventSink:
    """Buffer user-view events in an async queue and batch-write to the database."""

    def __init__(self, pool: Any, *, max_queue_size: int = 10_000) -> None:
        self._pool = pool
        self._queue: asyncio.Queue[ViewLog] = asyncio.Queue(maxsize=max_queue_size)

    async def start(self) -> None:
        """Launch the background batch-writer task."""
        asyncio.create_task(self._batch_writer())

    def enqueue(self, view: ViewLog) -> None:
        """Add a view event to the queue.

        Raises ``asyncio.QueueFull`` if the buffer is at capacity.
        """
        try:
            self._queue.put_nowait(view)
        except asyncio.QueueFull:
            RECSYS_VIEW_LOG_FAILURES_TOTAL.inc()
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _batch_writer(self) -> None:
        """Background task to batch-write user views without blocking the API."""
        while True:
            batch: list[ViewLog] = []
            try:
                # Wait for first item to arrive
                view = await self._queue.get()
                batch.append(view)

                # Collect more items for up to 1 second or until batch size 1000
                start_collect = time.time()
                while len(batch) < 1000 and (time.time() - start_collect) < 1.0:
                    try:
                        view = self._queue.get_nowait()
                        batch.append(view)
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.1)

                if batch:
                    async with self._pool.acquire() as conn:
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
                                v.eventdate or time.strftime("%Y-%m-%d"),
                            )
                            for v in batch
                        ]
                        await conn.executemany(query, values)
            except Exception as exc:
                RECSYS_VIEW_LOG_FAILURES_TOTAL.inc(len(batch) or 1)
                logger.exception("Failed to batch write user views: {}", exc)

            # Yield control to prevent CPU spinning if queue was empty
            await asyncio.sleep(0.5)
