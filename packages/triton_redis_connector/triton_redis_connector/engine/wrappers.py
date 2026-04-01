"""Framework wrappers that separate concurrency concerns from business logic.

Each wrapper adds ``stop: threading.Event`` and a ``variant`` property so the
pipeline registry can wire queues and manage lifecycle without the inner
(user-authored) node knowing about threading, processes, or the framework.

Wrappers optionally accept a ``MetricAggregator`` to collect per-call
throughput, latency, and error counters for parallel nodes.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from triton_redis_connector.engine.metric_aggregator import MetricAggregator  # noqa: TC001 — needed at runtime
from triton_redis_connector.engine.registry import ExecutionVariant


class DetachedThreadSource[TOut]:
    """Wraps an inline Source to run in a dedicated thread."""

    def __init__(  # noqa: D107
        self,
        inner: object,
        aggregator: MetricAggregator | None = None,
    ) -> None:
        self._inner = inner
        self.aggregator = aggregator
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_SOURCE

    def __call__(self) -> TOut | None:  # noqa: D102
        if self.aggregator is None:
            return self._inner()  # type: ignore[no-any-return, operator]
        t0 = time.monotonic()
        try:
            result: TOut | None = self._inner()  # type: ignore[operator]
        except Exception:
            self.aggregator.record_call((time.monotonic() - t0) * 1000.0, is_error=True)
            raise
        self.aggregator.record_call(
            (time.monotonic() - t0) * 1000.0,
            is_empty=(result is None),
        )
        return result


class DetachedProcessSource[TOut]:
    """Wraps an inline Source to run in a dedicated OS process."""

    def __init__(  # noqa: D107
        self,
        inner: object,
        aggregator: MetricAggregator | None = None,
    ) -> None:
        self._inner = inner
        self.aggregator = aggregator
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.PROCESS_SOURCE

    def __call__(self) -> TOut | None:  # noqa: D102
        if self.aggregator is None:
            return self._inner()  # type: ignore[no-any-return, operator]
        t0 = time.monotonic()
        try:
            result: TOut | None = self._inner()  # type: ignore[operator]
        except Exception:
            self.aggregator.record_call((time.monotonic() - t0) * 1000.0, is_error=True)
            raise
        self.aggregator.record_call(
            (time.monotonic() - t0) * 1000.0,
            is_empty=(result is None),
        )
        return result


class DetachedThreadProcessor[TIn, TOut]:
    """Wraps an inline Processor to run in a dedicated thread."""

    def __init__(  # noqa: D107
        self,
        inner: object,
        aggregator: MetricAggregator | None = None,
    ) -> None:
        self._inner = inner
        self.aggregator = aggregator
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_PROCESSOR

    def __call__(self, event: TIn) -> TOut | None:  # noqa: D102
        if self.aggregator is None:
            return self._inner(event)  # type: ignore[no-any-return, operator]
        t0 = time.monotonic()
        try:
            result: TOut | None = self._inner(event)  # type: ignore[operator]
        except Exception:
            self.aggregator.record_call((time.monotonic() - t0) * 1000.0, is_error=True)
            raise
        self.aggregator.record_call(
            (time.monotonic() - t0) * 1000.0,
            is_empty=(result is None),
        )
        return result


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
    ) -> None:
        self._inner = inner
        self.aggregator = aggregator
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_POOL_PROCESSOR

    def __call__(self, event: TIn) -> TOut | None:  # noqa: D102
        if self.aggregator is None:
            return self._inner(event)  # type: ignore[no-any-return, operator]
        t0 = time.monotonic()
        try:
            result: TOut | None = self._inner(event)  # type: ignore[operator]
        except Exception:
            self.aggregator.record_call((time.monotonic() - t0) * 1000.0, is_error=True)
            raise
        self.aggregator.record_call(
            (time.monotonic() - t0) * 1000.0,
            is_empty=(result is None),
        )
        return result


class MultiProcessProcessor[TIn, TOut]:
    """Wraps a single Processor instance for process-pool execution.

    Same pattern as ``ThreadedProcessor``: the framework creates N instances
    via a factory, each living in its own OS process with isolated state.
    """

    def __init__(  # noqa: D107
        self,
        inner: object,
        aggregator: MetricAggregator | None = None,
    ) -> None:
        self._inner = inner
        self.aggregator = aggregator
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.PROCESS_POOL_PROCESSOR

    def __call__(self, event: TIn) -> TOut | None:  # noqa: D102
        if self.aggregator is None:
            return self._inner(event)  # type: ignore[no-any-return, operator]
        t0 = time.monotonic()
        try:
            result: TOut | None = self._inner(event)  # type: ignore[operator]
        except Exception:
            self.aggregator.record_call((time.monotonic() - t0) * 1000.0, is_error=True)
            raise
        self.aggregator.record_call(
            (time.monotonic() - t0) * 1000.0,
            is_empty=(result is None),
        )
        return result


class DetachedThreadSink[TIn]:
    """Wraps an inline Sink to run in a dedicated thread."""

    def __init__(  # noqa: D107
        self,
        inner: object,
        aggregator: MetricAggregator | None = None,
    ) -> None:
        self._inner = inner
        self.aggregator = aggregator
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_SINK

    def __call__(self, event: TIn) -> Any | None:  # noqa: ANN401, D102
        if self.aggregator is None:
            return self._inner(event)  # type: ignore[operator]
        t0 = time.monotonic()
        try:
            result: Any = self._inner(event)  # type: ignore[operator]
        except Exception:
            self.aggregator.record_call((time.monotonic() - t0) * 1000.0, is_error=True)
            raise
        self.aggregator.record_call((time.monotonic() - t0) * 1000.0)
        return result
