"""Framework wrappers that separate concurrency concerns from business logic.

Each wrapper adds ``stop: threading.Event`` and a ``variant`` property so the
pipeline registry can wire queues and manage lifecycle without the inner
(user-authored) node knowing about threading, processes, or the framework.
"""

from __future__ import annotations

import threading
from typing import Any

from triton_redis_connector.engine.registry import ExecutionVariant


class DetachedThreadSource[TOut]:
    """Wraps an inline Source to run in a dedicated thread."""

    def __init__(self, inner: object) -> None:  # noqa: D107
        self._inner = inner
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_SOURCE

    def __call__(self) -> TOut | None:  # noqa: D102
        return self._inner()  # type: ignore[no-any-return, operator]


class DetachedProcessSource[TOut]:
    """Wraps an inline Source to run in a dedicated OS process."""

    def __init__(self, inner: object) -> None:  # noqa: D107
        self._inner = inner
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.PROCESS_SOURCE

    def __call__(self) -> TOut | None:  # noqa: D102
        return self._inner()  # type: ignore[no-any-return, operator]


class DetachedThreadProcessor[TIn, TOut]:
    """Wraps an inline Processor to run in a dedicated thread."""

    def __init__(self, inner: object) -> None:  # noqa: D107
        self._inner = inner
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_PROCESSOR

    def __call__(self, event: TIn) -> TOut | None:  # noqa: D102
        return self._inner(event)  # type: ignore[no-any-return, operator]


class ThreadedProcessor[TIn, TOut]:
    """Wraps a single Processor instance for thread-pool execution.

    The framework creates N ``ThreadedProcessor`` instances (one per worker)
    by calling the user-supplied factory N times and wrapping each result.
    Each wrapper instance holds its own independent inner processor.
    """

    def __init__(self, inner: object) -> None:  # noqa: D107
        self._inner = inner
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_POOL_PROCESSOR

    def __call__(self, event: TIn) -> TOut | None:  # noqa: D102
        return self._inner(event)  # type: ignore[no-any-return, operator]


class MultiProcessProcessor[TIn, TOut]:
    """Wraps a single Processor instance for process-pool execution.

    Same pattern as ``ThreadedProcessor``: the framework creates N instances
    via a factory, each living in its own OS process with isolated state.
    """

    def __init__(self, inner: object) -> None:  # noqa: D107
        self._inner = inner
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.PROCESS_POOL_PROCESSOR

    def __call__(self, event: TIn) -> TOut | None:  # noqa: D102
        return self._inner(event)  # type: ignore[no-any-return, operator]


class DetachedThreadSink[TIn]:
    """Wraps an inline Sink to run in a dedicated thread."""

    def __init__(self, inner: object) -> None:  # noqa: D107
        self._inner = inner
        self.stop = threading.Event()

    @property
    def variant(self) -> ExecutionVariant:  # noqa: D102
        return ExecutionVariant.THREAD_SINK

    def __call__(self, event: TIn) -> Any | None:  # noqa: ANN401, D102
        return self._inner(event)  # type: ignore[operator]
