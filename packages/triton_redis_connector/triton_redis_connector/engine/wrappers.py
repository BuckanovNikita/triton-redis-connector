"""Framework wrappers that separate concurrency concerns from business logic.

Each wrapper adds ``stop: threading.Event`` and a ``variant`` property so the
pipeline registry can wire queues and manage lifecycle without the inner
(user-authored) node knowing about threading, processes, or the framework.

Wrappers optionally accept a ``MetricAggregator`` to collect per-call
throughput, latency, and error counters for parallel nodes.

When ``metric_recorder`` is provided, wrappers extract Counter/Gauge/GroupBy
metrics declared on result event classes and record them to the buffer.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from triton_redis_connector.engine.metric_aggregator import (
    MetricAggregator,
    extract_and_record_metrics,
)
from triton_redis_connector.engine.registry import ExecutionVariant

if TYPE_CHECKING:
    from collections.abc import Callable

    from triton_redis_connector.protocols import MetricRecorder


def _call_with_metrics(
    inner: Callable[..., Any],
    event: object | None,
    aggregator: MetricAggregator | None,
    metric_recorder: MetricRecorder | None,
    component_name: str,
) -> object:
    """Shared call logic: timing, aggregation, event-level metric extraction."""
    if aggregator is None and metric_recorder is None:
        return inner(event) if event is not None else inner()

    t0 = time.monotonic()
    try:
        result: object = inner(event) if event is not None else inner()
    except Exception:
        if aggregator is not None:
            aggregator.record_call((time.monotonic() - t0) * 1000.0, is_error=True)
        raise
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    if aggregator is not None:
        aggregator.record_call(elapsed_ms, is_empty=(result is None))

    if metric_recorder is not None and result is not None:
        extract_and_record_metrics(result, metric_recorder, component_name, elapsed_ms)

    return result


class DetachedThreadSource[TOut]:
    """Wraps a source callable to run in a dedicated thread."""

    def __init__(  # noqa: D107
        self,
        inner: Callable[[], TOut | None],
        aggregator: MetricAggregator | None = None,
        metric_recorder: MetricRecorder | None = None,
        component_name: str = "",
    ) -> None:
        self._inner: Callable[[], TOut | None] = inner
        self.aggregator = aggregator
        self._metric_recorder = metric_recorder
        self._component_name = component_name
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_SOURCE

    def __call__(self) -> TOut | None:  # noqa: D102
        return _call_with_metrics(  # type: ignore[return-value]
            self._inner, None, self.aggregator, self._metric_recorder, self._component_name
        )


class DetachedProcessSource[TOut]:
    """Wraps a source callable to run in a dedicated OS process."""

    def __init__(  # noqa: D107
        self,
        inner: Callable[[], TOut | None],
        aggregator: MetricAggregator | None = None,
        metric_recorder: MetricRecorder | None = None,
        component_name: str = "",
    ) -> None:
        self._inner: Callable[[], TOut | None] = inner
        self.aggregator = aggregator
        self._metric_recorder = metric_recorder
        self._component_name = component_name
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.PROCESS_SOURCE

    def __call__(self) -> TOut | None:  # noqa: D102
        return _call_with_metrics(  # type: ignore[return-value]
            self._inner, None, self.aggregator, self._metric_recorder, self._component_name
        )


class DetachedThreadProcessor[TIn, TOut]:
    """Wraps a processor callable to run in a dedicated thread."""

    def __init__(  # noqa: D107
        self,
        inner: Callable[[TIn], TOut | None],
        aggregator: MetricAggregator | None = None,
        metric_recorder: MetricRecorder | None = None,
        component_name: str = "",
    ) -> None:
        self._inner: Callable[[TIn], TOut | None] = inner
        self.aggregator = aggregator
        self._metric_recorder = metric_recorder
        self._component_name = component_name
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_PROCESSOR

    def __call__(self, event: TIn) -> TOut | None:  # noqa: D102
        return _call_with_metrics(  # type: ignore[return-value]
            self._inner, event, self.aggregator, self._metric_recorder, self._component_name
        )


class ThreadedProcessor[TIn, TOut]:
    """Wraps a single processor callable for thread-pool execution.

    The framework creates N ``ThreadedProcessor`` instances (one per worker)
    by calling the user-supplied factory N times and wrapping each result.
    Each wrapper instance holds its own independent inner processor.
    """

    def __init__(  # noqa: D107
        self,
        inner: Callable[[TIn], TOut | None],
        aggregator: MetricAggregator | None = None,
        metric_recorder: MetricRecorder | None = None,
        component_name: str = "",
    ) -> None:
        self._inner: Callable[[TIn], TOut | None] = inner
        self.aggregator = aggregator
        self._metric_recorder = metric_recorder
        self._component_name = component_name
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_POOL_PROCESSOR

    def __call__(self, event: TIn) -> TOut | None:  # noqa: D102
        return _call_with_metrics(  # type: ignore[return-value]
            self._inner, event, self.aggregator, self._metric_recorder, self._component_name
        )


class MultiProcessProcessor[TIn, TOut]:
    """Wraps a single processor callable for process-pool execution.

    Same pattern as ``ThreadedProcessor``: the framework creates N instances
    via a factory, each living in its own OS process with isolated state.
    """

    def __init__(  # noqa: D107
        self,
        inner: Callable[[TIn], TOut | None],
        aggregator: MetricAggregator | None = None,
        metric_recorder: MetricRecorder | None = None,
        component_name: str = "",
    ) -> None:
        self._inner: Callable[[TIn], TOut | None] = inner
        self.aggregator = aggregator
        self._metric_recorder = metric_recorder
        self._component_name = component_name
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.PROCESS_POOL_PROCESSOR

    def __call__(self, event: TIn) -> TOut | None:  # noqa: D102
        return _call_with_metrics(  # type: ignore[return-value]
            self._inner, event, self.aggregator, self._metric_recorder, self._component_name
        )


class DetachedThreadSink[TIn]:
    """Wraps a sink callable to run in a dedicated thread."""

    def __init__(  # noqa: D107
        self,
        inner: Callable[[TIn], Any | None],
        aggregator: MetricAggregator | None = None,
        metric_recorder: MetricRecorder | None = None,
        component_name: str = "",
    ) -> None:
        self._inner: Callable[[TIn], Any | None] = inner
        self.aggregator = aggregator
        self._metric_recorder = metric_recorder
        self._component_name = component_name
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_SINK

    def __call__(self, event: TIn) -> Any | None:  # noqa: ANN401, D102
        return _call_with_metrics(self._inner, event, self.aggregator, self._metric_recorder, self._component_name)
