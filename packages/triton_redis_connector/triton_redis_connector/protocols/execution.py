"""Execution variant protocols and concrete base classes.

Six protocol classes define the contract for each parallel execution variant.
Six concrete base classes provide default implementations (stop event, variant
property) for user-defined pipeline nodes.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, TypeVar, runtime_checkable

TOut_co = TypeVar("TOut_co", covariant=True)
TIn_contra = TypeVar("TIn_contra", contravariant=True)


@runtime_checkable
class ThreadSource(Protocol[TOut_co]):
    """Threaded source. Runs in a dedicated thread, may block internally.

    Framework creates stop Event, assigns before starting the thread,
    calls stop.set() at shutdown. Output goes to framework-wired queue.
    """

    stop: threading.Event

    def __call__(self) -> TOut_co | None:
        """Produce an event or return None if nothing available."""
        ...


@runtime_checkable
class ProcessSource(Protocol[TOut_co]):
    """Process source. Runs in a dedicated OS process.

    Same contract as ThreadSource but in a separate process.
    stop is a multiprocessing.Event (duck-typed as threading.Event).
    """

    stop: threading.Event

    def __call__(self) -> TOut_co | None:
        """Produce an event or return None if nothing available."""
        ...


@runtime_checkable
class ThreadProcessor(Protocol[TIn_contra, TOut_co]):
    """Threaded processor. Runs in a dedicated thread.

    Owns input_queue and output_queue, wired by framework.
    """

    stop: threading.Event

    def __call__(self, event: TIn_contra) -> TOut_co | None:
        """Transform an event or return None to filter it out."""
        ...


@runtime_checkable
class ThreadPoolProcessor(Protocol[TIn_contra, TOut_co]):
    """Thread pool processor. N threads, shared input queue.

    Framework creates N instances (one per worker) and shared queues.
    """

    stop: threading.Event

    def __call__(self, event: TIn_contra) -> TOut_co | None:
        """Transform an event or return None to filter it out."""
        ...


@runtime_checkable
class ProcessPoolProcessor(Protocol[TIn_contra, TOut_co]):
    """Process pool processor. N processes, shared input queue.

    Framework creates N instances via factory. True OS-level parallelism.
    """

    stop: threading.Event

    def __call__(self, event: TIn_contra) -> TOut_co | None:
        """Transform an event or return None to filter it out."""
        ...


@runtime_checkable
class ThreadSink(Protocol[TIn_contra]):
    """Threaded sink. Runs in a dedicated thread, owns input queue.

    Framework wires input queue. Can optionally return metric events.
    """

    stop: threading.Event

    def __call__(self, event: TIn_contra) -> Any | None:  # noqa: ANN401
        """Consume an event and optionally return a metric event."""
        ...


class BaseThreadSource[TOut]:
    """Threaded source with dedicated thread. Subclass and override ``__call__``."""

    def __init__(self) -> None:  # noqa: D107
        self.stop = threading.Event()

    @property
    def variant(self) -> str:  # noqa: D102
        return "thread_source"

    def __call__(self) -> TOut | None:
        """Produce an event or return None if nothing available."""
        raise NotImplementedError


class BaseProcessSource[TOut]:
    """Process source with dedicated OS process. Subclass and override ``__call__``."""

    def __init__(self) -> None:  # noqa: D107
        self.stop = threading.Event()

    @property
    def variant(self) -> str:  # noqa: D102
        return "process_source"

    def __call__(self) -> TOut | None:
        """Produce an event or return None if nothing available."""
        raise NotImplementedError


class BaseThreadProcessor[TIn, TOut]:
    """Dedicated-thread processor. Subclass and override ``__call__``."""

    def __init__(self) -> None:  # noqa: D107
        self.stop = threading.Event()

    @property
    def variant(self) -> str:  # noqa: D102
        return "thread_processor"

    def __call__(self, event: TIn) -> TOut | None:
        """Transform an event or return None to filter it out."""
        raise NotImplementedError


class BaseThreadPoolProcessor[TIn, TOut]:
    """Thread-pool processor (N workers, shared queue). Subclass and override ``__call__``."""

    def __init__(self) -> None:  # noqa: D107
        self.stop = threading.Event()

    @property
    def variant(self) -> str:  # noqa: D102
        return "thread_pool_processor"

    def __call__(self, event: TIn) -> TOut | None:
        """Transform an event or return None to filter it out."""
        raise NotImplementedError


class BaseProcessPoolProcessor[TIn, TOut]:
    """Process-pool processor (N workers, true OS parallelism). Subclass and override ``__call__``."""

    def __init__(self) -> None:  # noqa: D107
        self.stop = threading.Event()

    @property
    def variant(self) -> str:  # noqa: D102
        return "process_pool_processor"

    def __call__(self, event: TIn) -> TOut | None:
        """Transform an event or return None to filter it out."""
        raise NotImplementedError


class BaseThreadSink[TIn]:
    """Threaded sink with dedicated thread. Subclass and override ``__call__``."""

    def __init__(self) -> None:  # noqa: D107
        self.stop = threading.Event()

    @property
    def variant(self) -> str:  # noqa: D102
        return "thread_sink"

    def __call__(self, event: TIn) -> Any | None:  # noqa: ANN401
        """Consume an event and optionally return a metric event."""
        raise NotImplementedError
