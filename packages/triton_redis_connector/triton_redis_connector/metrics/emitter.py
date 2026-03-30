"""Periodic emitter that flushes MetricBuffer to a Redis stream.

Runs in a background daemon thread, draining buffered metrics at a
configurable interval via XADD.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from redis import Redis

    from triton_redis_connector.metrics.buffer import MetricBuffer


class MetricEmitter:
    """Background daemon that periodically flushes MetricBuffer to Redis.

    Each flush cycle atomically swaps the buffer, serializes all MetricEvent
    instances, and writes them to the configured Redis stream via XADD.
    """

    def __init__(  # noqa: D107
        self,
        buffer: MetricBuffer,
        client: Redis[bytes],
        stream: str,
        interval_s: float,
    ) -> None:
        self._buffer = buffer
        self._client = client
        self._stream = stream
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background emission thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="metric-emitter",
            daemon=True,
        )
        self._thread.start()
        logger.info("MetricEmitter started (interval={}s, stream={})", self._interval_s, self._stream)

    def stop(self) -> None:
        """Signal the emission thread to stop and wait for it to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s * 2)
            self._thread = None
        logger.info("MetricEmitter stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=self._interval_s)
            self._flush_to_redis()

    def _flush_to_redis(self) -> None:
        events = self._buffer.flush()
        if not events:
            return
        for event in events:
            fields = {k: str(v) for k, v in event.model_dump().items()}
            try:
                self._client.xadd(  # pyright: ignore[reportUnknownMemberType]
                    name=self._stream,
                    fields=fields,
                    maxlen=10_000,
                    approximate=True,
                )
            except ConnectionError:
                logger.error(
                    "Failed to XADD metric to stream {}: measure={}",
                    self._stream,
                    event.measure,
                )
        logger.debug("Flushed {} metric events to {}", len(events), self._stream)
