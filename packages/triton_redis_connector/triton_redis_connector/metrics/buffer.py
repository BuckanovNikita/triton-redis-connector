"""Thread-safe buffer for MetricEvent instances.

Accumulates metrics from pipeline components and provides atomic flush
for periodic emission to Redis.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from triton_redis_connector.metrics.collector import MetricEvent


class MetricBuffer:
    """Thread-safe accumulator for MetricEvent instances.

    Multiple producers (sources, processors, sinks) call record() concurrently.
    MetricEmitter calls flush() periodically to drain all buffered events.

    Accepts ``object`` in record() so the engine layer can pass MetricEvent
    instances via the MetricRecorder protocol without importing the type.
    """

    def __init__(self) -> None:  # noqa: D107
        self._lock = threading.Lock()
        self._buffer: list[MetricEvent] = []

    def record(self, event: object) -> None:
        """Append a metric event to the buffer under lock."""
        with self._lock:
            self._buffer.append(event)  # type: ignore[arg-type]

    def flush(self) -> list[MetricEvent]:
        """Atomically swap the buffer and return all accumulated events."""
        with self._lock:
            events = self._buffer
            self._buffer = []
        return events
