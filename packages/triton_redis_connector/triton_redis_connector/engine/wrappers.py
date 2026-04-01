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
from typing import Any

from triton_redis_connector.engine.metric_aggregator import (
    MetricAggregator,
    extract_event_counters,
    extract_field_gauges,
    extract_field_group_bys,
    extract_field_value,
)
from triton_redis_connector.engine.registry import ExecutionVariant
from triton_redis_connector.protocols import MetricRecorder  # noqa: TC001 — needed at runtime


def _make_metric(component_name: str, measure: str, value: float) -> object:
    """Create a MetricEvent via lazy import to avoid engine->metrics layer violation."""
    from triton_redis_connector.metrics.collector import MetricEvent  # noqa: PLC0415

    return MetricEvent(component_name=component_name, measure=measure, value=value)


def _extract_and_record_metrics(
    result: object,
    recorder: MetricRecorder,
    component_name: str,
    elapsed_ms: float,
) -> None:
    """Extract Counter/Gauge/GroupBy metrics from a result event and record them."""
    result_type = type(result)

    for counter in extract_event_counters(result_type):
        recorder.record(_make_metric(component_name, counter.name, 1.0))

    for field_name, gauge in extract_field_gauges(result_type):
        raw = extract_field_value(result, field_name)
        if isinstance(raw, (int, float)):
            recorder.record(_make_metric(component_name, gauge.name, float(raw)))

    for field_name, group_by in extract_field_group_bys(result_type):
        raw = extract_field_value(result, field_name)
        if raw is not None:
            recorder.record(_make_metric(component_name, f"{group_by.name}:{raw!s}", 1.0))

    recorder.record(_make_metric(component_name, "node_processing_ms", elapsed_ms))


def _call_with_metrics(
    inner: object,
    event: object | None,
    aggregator: MetricAggregator | None,
    metric_recorder: MetricRecorder | None,
    component_name: str,
) -> object:
    """Shared call logic: timing, aggregation, event-level metric extraction."""
    if aggregator is None and metric_recorder is None:
        return inner(event) if event is not None else inner()  # type: ignore[operator]

    t0 = time.monotonic()
    try:
        result: object = inner(event) if event is not None else inner()  # type: ignore[operator]
    except Exception:
        if aggregator is not None:
            aggregator.record_call((time.monotonic() - t0) * 1000.0, is_error=True)
        raise
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    if aggregator is not None:
        aggregator.record_call(elapsed_ms, is_empty=(result is None))

    if metric_recorder is not None and result is not None:
        _extract_and_record_metrics(result, metric_recorder, component_name, elapsed_ms)

    return result


class DetachedThreadSource[TOut]:
    """Wraps an inline Source to run in a dedicated thread."""

    def __init__(  # noqa: D107
        self,
        inner: object,
        aggregator: MetricAggregator | None = None,
        metric_recorder: MetricRecorder | None = None,
        component_name: str = "",
    ) -> None:
        self._inner = inner
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
    """Wraps an inline Source to run in a dedicated OS process."""

    def __init__(  # noqa: D107
        self,
        inner: object,
        aggregator: MetricAggregator | None = None,
        metric_recorder: MetricRecorder | None = None,
        component_name: str = "",
    ) -> None:
        self._inner = inner
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
    """Wraps an inline Processor to run in a dedicated thread."""

    def __init__(  # noqa: D107
        self,
        inner: object,
        aggregator: MetricAggregator | None = None,
        metric_recorder: MetricRecorder | None = None,
        component_name: str = "",
    ) -> None:
        self._inner = inner
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
    """Wraps a single Processor instance for thread-pool execution.

    The framework creates N ``ThreadedProcessor`` instances (one per worker)
    by calling the user-supplied factory N times and wrapping each result.
    Each wrapper instance holds its own independent inner processor.
    """

    def __init__(  # noqa: D107
        self,
        inner: object,
        aggregator: MetricAggregator | None = None,
        metric_recorder: MetricRecorder | None = None,
        component_name: str = "",
    ) -> None:
        self._inner = inner
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
    """Wraps a single Processor instance for process-pool execution.

    Same pattern as ``ThreadedProcessor``: the framework creates N instances
    via a factory, each living in its own OS process with isolated state.
    """

    def __init__(  # noqa: D107
        self,
        inner: object,
        aggregator: MetricAggregator | None = None,
        metric_recorder: MetricRecorder | None = None,
        component_name: str = "",
    ) -> None:
        self._inner = inner
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
    """Wraps an inline Sink to run in a dedicated thread."""

    def __init__(  # noqa: D107
        self,
        inner: object,
        aggregator: MetricAggregator | None = None,
        metric_recorder: MetricRecorder | None = None,
        component_name: str = "",
    ) -> None:
        self._inner = inner
        self.aggregator = aggregator
        self._metric_recorder = metric_recorder
        self._component_name = component_name
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_SINK

    def __call__(self, event: TIn) -> Any | None:  # noqa: ANN401, D102
        return _call_with_metrics(self._inner, event, self.aggregator, self._metric_recorder, self._component_name)
