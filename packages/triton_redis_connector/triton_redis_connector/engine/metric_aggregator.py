"""Thread-safe metric aggregation for pipeline nodes.

MetricAggregator wraps node calls to collect per-node throughput,
latency, and error counters. Tracked is an Annotated marker that
declares a node should be instrumented automatically by wrappers.

Usage::

    from typing import Annotated
    from triton_redis_connector.engine.metric_aggregator import Tracked

    InferenceType = Annotated[InferenceWorker, Tracked("inference")]
"""

from __future__ import annotations

import threading
import typing
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Tracked:
    """Annotated marker declaring a node should collect metrics.

    Usage: ``Annotated[MyProcessor, Tracked("my-processor")]``

    The name field is optional — when empty, the registry node name is used.
    """

    name: str = ""


@dataclass(frozen=True, slots=True)
class NodeMetricSnapshot:
    """Immutable snapshot of a node's accumulated metrics."""

    call_count: int
    error_count: int
    empty_count: int
    total_processing_ms: float


class MetricAggregator:
    """Thread-safe counters for a single pipeline node.

    Wrappers call ``record_call`` after each ``__call__`` invocation.
    ``Application._log_stats`` reads ``snapshot()`` and ``interval_calls()``
    periodically to emit per-node metrics.
    """

    __slots__ = (
        "_call_count",
        "_empty_count",
        "_error_count",
        "_lock",
        "_prev_call_count",
        "_total_ms",
    )

    def __init__(self) -> None:  # noqa: D107
        self._lock = threading.Lock()
        self._call_count: int = 0
        self._error_count: int = 0
        self._empty_count: int = 0
        self._total_ms: float = 0.0
        self._prev_call_count: int = 0

    def record_call(
        self,
        elapsed_ms: float,
        *,
        is_error: bool = False,
        is_empty: bool = False,
    ) -> None:
        """Record one call with timing and optional error/empty flags."""
        with self._lock:
            self._call_count += 1
            self._total_ms += elapsed_ms
            if is_error:
                self._error_count += 1
            if is_empty:
                self._empty_count += 1

    def snapshot(self) -> NodeMetricSnapshot:
        """Return an immutable snapshot of all accumulated counters."""
        with self._lock:
            return NodeMetricSnapshot(
                call_count=self._call_count,
                error_count=self._error_count,
                empty_count=self._empty_count,
                total_processing_ms=self._total_ms,
            )

    def interval_calls(self) -> int:
        """Return calls since last ``interval_calls()`` and advance the window."""
        with self._lock:
            delta = self._call_count - self._prev_call_count
            self._prev_call_count = self._call_count
            return delta


@runtime_checkable
class InstrumentedNode(Protocol):
    """Node wrapper that exposes a ``MetricAggregator`` for stats collection.

    Used by ``PipelineRegistry`` to discover aggregators without ``getattr``.
    """

    aggregator: MetricAggregator | None


def extract_tracked(annotation: Any) -> Tracked | None:  # noqa: ANN401
    """Extract a ``Tracked`` marker from ``Annotated`` metadata.

    Returns ``None`` if the annotation is not ``Annotated`` or has no
    ``Tracked`` marker.
    """
    args = typing.get_args(annotation)
    for arg in args:
        if isinstance(arg, Tracked):
            return arg
    return None
