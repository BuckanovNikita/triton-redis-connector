"""Queue bus for inter-node communication in the processing graph.

Manages named queues with configurable maxsize and backpressure
via drop-and-log policy.
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any

from loguru import logger  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from triton_redis_connector.config.schema import QueueConfig


class QueueOverflow:
    """Returned when an item is dropped due to a full queue."""

    __slots__ = ("queue_name",)

    def __init__(self, queue_name: str) -> None:  # noqa: D107
        self.queue_name = queue_name


class QueueBus:
    """Manages named queues for the processing graph.

    All queues are created with a configurable maxsize. When a queue
    is full, the put operation drops the item, logs a warning, and
    returns a QueueOverflow indicator.
    """

    def __init__(self, config: QueueConfig) -> None:  # noqa: D107
        self._config = config
        self._queues: dict[str, queue.Queue[Any]] = {}

    def get_queue(self, name: str, maxsize: int | None = None) -> queue.Queue[Any]:
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

    def put(self, name: str, item: Any) -> QueueOverflow | None:  # noqa: ANN401
        """Put an item on a named queue with backpressure handling.

        If the queue is full, drops the item and returns a
        QueueOverflow indicator. Returns None on success.
        """
        q = self.get_queue(name)
        try:
            q.put_nowait(item)
        except queue.Full:
            logger.warning("Queue {} is full (maxsize={}), dropping item", name, q.maxsize)
            return QueueOverflow(queue_name=name)
        return None

    def get(self, name: str, timeout: float | None = None) -> Any:  # noqa: ANN401
        """Get an item from a named queue.

        Raises queue.Empty if the queue is empty and timeout expires.
        """
        q = self.get_queue(name)
        return q.get(timeout=timeout)

    @property
    def queue_names(self) -> list[str]:
        """Return the names of all created queues."""
        return list(self._queues.keys())
