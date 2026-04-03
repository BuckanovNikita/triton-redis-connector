"""Periodic emitter that flushes MetricBuffer to a Redis stream.

Runs in a background daemon thread, draining buffered metrics at a
configurable interval via XADD. Supports a zero-value guarantee:
registered metric names that are absent in a flush cycle are emitted
with value 0.0 so monitoring dashboards always have data.

The ``component_name`` field is set to ``APP_NAME`` (env) or hostname
at flush time. The ``measure`` field is the metric's own unique name
without any app/host prefix.
"""

from __future__ import annotations

import socket
import threading
from typing import TYPE_CHECKING

from loguru import logger
from redis import exceptions as redis_exceptions

from typed_flow.metrics.collector import MetricEvent

if TYPE_CHECKING:
    from redis import Redis

    from typed_flow.metrics.buffer import MetricBuffer


def _sanitize_prefix(raw: str) -> str:
    """Normalize a prefix string: replace non-alphanumeric chars with ``_``."""
    return "".join(c if c.isalnum() else "_" for c in raw).strip("_")


class MetricEmitter:
    """Background daemon that periodically flushes MetricBuffer to Redis.

    Each flush cycle atomically swaps the buffer, serializes all MetricEvent
    instances, and writes them to the configured Redis stream via XADD.
    After flushing real events, emits zero-value MetricEvents for any
    known metric names not seen in this cycle.

    The ``component_name`` is set to ``app_name`` or hostname at emit time.
    Measure names are emitted as-is without prefixing.
    """

    def __init__(  # noqa: PLR0913
        self,
        buffer: MetricBuffer,
        client: Redis[bytes],
        stream: str,
        interval_s: float,
        known_metrics: set[str] | None = None,
        app_name: str = "",
    ) -> None:
        """Initialize emitter with buffer, Redis client, stream name, and interval."""
        self._buffer = buffer
        self._client = client
        self._stream = stream
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._known_metrics: set[str] = known_metrics.copy() if known_metrics is not None else set()
        self._component_name = _sanitize_prefix(app_name) if app_name else _sanitize_prefix(socket.gethostname())

    @property
    def component_name(self) -> str:
        """Return the resolved component name (APP_NAME or hostname)."""
        return self._component_name

    def register_metric(self, name: str) -> None:
        """Register a metric name for zero-value guarantee."""
        self._known_metrics.add(name)

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
        logger.info(
            "MetricEmitter started (interval={}s, stream={}, component={})",
            self._interval_s,
            self._stream,
            self._component_name,
        )

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
        seen_this_cycle: set[str] = set()

        for event in events:
            seen_this_cycle.add(event.measure)
            emitted = MetricEvent(
                component_name=self._component_name,
                measure=event.measure,
                value=event.value,
            )
            self._xadd_event(emitted)

        for name in self._known_metrics:
            if name not in seen_this_cycle:
                zero_event = MetricEvent(
                    component_name=self._component_name,
                    measure=name,
                    value=0.0,
                )
                self._xadd_event(zero_event)

        total = len(events) + len(self._known_metrics - seen_this_cycle)
        if total > 0:
            logger.debug("Flushed {} metric events to {}", total, self._stream)

    def _xadd_event(self, event: MetricEvent) -> None:
        """Write a single MetricEvent to Redis via XADD."""
        fields = {k: str(v) for k, v in event.model_dump().items()}
        try:
            self._client.xadd(
                name=self._stream,
                fields=fields,
                maxlen=10_000,
                approximate=True,
            )
        except redis_exceptions.ConnectionError:
            logger.error(
                "Failed to XADD metric to stream {}: measure={}",
                self._stream,
                event.measure,
            )
