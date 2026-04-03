"""Queue bus for inter-node communication in the processing graph.

Manages named queues with configurable maxsize and backpressure
via drop-and-log policy.
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from typed_flow.config.schema import QueueConfig


class QueueOverflow:
    """Returned when an item is dropped due to a full queue."""

    __slots__ = ("queue_name",)

    def __init__(self, queue_name: str) -> None:
        """Initialize with the name of the overflowed queue."""
        self.queue_name = queue_name


class QueueBus:
    """Manages named queues for the processing graph.

    All queues are created with a configurable maxsize. When a queue
    is full, the put operation drops the item, logs a warning, and
    returns a QueueOverflow indicator.
    """

    def __init__(self, config: QueueConfig) -> None:
        """Initialize queue bus with queue configuration."""
        self._config = config
        self._queues: dict[str, queue.Queue[object]] = {}
        self._drop_counts: dict[str, int] = {}

    def get_queue(self, name: str, maxsize: int | None = None) -> queue.Queue[object]:
        """Get or create a named queue.

        If the queue already exists, returns it regardless of the
        maxsize parameter. New queues use the provided maxsize or
        fall back to the config default.
        """
        if name not in self._queues:
            effective_maxsize = maxsize if maxsize is not None else self._config.maxsize
            self._queues[name] = queue.Queue(maxsize=effective_maxsize)
            logger.debug("Created queue {} with maxsize={}", name, effective_maxsize)
        return self._queues[name]

    PUT_TIMEOUT_S: float = 5.0

    def put(self, name: str, item: object) -> QueueOverflow | None:
        """Put an item on a named queue with backpressure handling.

        Blocks up to PUT_TIMEOUT_S when the queue is full, applying
        backpressure to the producer. If still full after timeout,
        drops the item and returns a QueueOverflow indicator.
        Returns None on success.
        """
        q = self.get_queue(name)
        try:
            q.put(item, timeout=self.PUT_TIMEOUT_S)
        except queue.Full:
            self._drop_counts[name] = self._drop_counts.get(name, 0) + 1
            logger.warning(
                "Queue {} is full (maxsize={}), dropping item after {}s backpressure",
                name,
                q.maxsize,
                self.PUT_TIMEOUT_S,
            )
            return QueueOverflow(queue_name=name)
        return None

    MAX_TIMEOUT_S: float = 5.0

    def get(self, name: str, timeout: float = 0.1) -> object:
        """Get an item from a named queue with bounded timeout.

        Timeout is capped at MAX_TIMEOUT_S to prevent deadlocks.
        Raises queue.Empty if the queue is empty and timeout expires.
        """
        q = self.get_queue(name)
        bounded = min(timeout, self.MAX_TIMEOUT_S)
        return q.get(timeout=bounded)

    def get_nowait(self, name: str) -> object:
        """Get an item from a named queue without blocking.

        Raises queue.Empty if the queue is empty.
        """
        q = self.get_queue(name)
        return q.get_nowait()

    def queue_depths(self) -> dict[str, int]:
        """Return approximate current depth of each managed queue."""
        return {name: q.qsize() for name, q in self._queues.items()}

    def drop_counts(self) -> dict[str, int]:
        """Return cumulative drop count per queue name."""
        return dict(self._drop_counts)

    @property
    def queue_names(self) -> list[str]:
        """Return the names of all created queues."""
        return list(self._queues.keys())
