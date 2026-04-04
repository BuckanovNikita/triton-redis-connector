"""Execution variant protocols and concrete base classes.

Six protocol classes define the contract for each parallel execution variant.
Six concrete base classes provide default implementations (stop event, variant
property) for user-defined pipeline nodes.
"""

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


@runtime_checkable
class ProcessSink(Protocol[TIn_contra]):
    """Process sink. Runs in a dedicated OS process, owns input queue.

    Same contract as ThreadSink but in a separate process.
    stop is a multiprocessing.Event (duck-typed as threading.Event).
    """

    stop: threading.Event

    def __call__(self, event: TIn_contra) -> Any | None:  # noqa: ANN401
        """Consume an event and optionally return a metric event."""
        ...


class BaseThreadSource[TOut]:
    """Threaded source with dedicated thread. Subclass and override ``__call__``."""

    def __init__(self) -> None:
        """Initialize stop event for thread lifecycle management."""
        self.stop = threading.Event()

    @property
    def variant(self) -> str:
        """Return the execution variant identifier."""
        return "thread_source"

    def __call__(self) -> TOut | None:
        """Produce an event or return None if nothing available."""
        raise NotImplementedError


class BaseProcessSource[TOut]:
    """Process source with dedicated OS process. Subclass and override ``__call__``."""

    def __init__(self) -> None:
        """Initialize stop event for process lifecycle management."""
        self.stop = threading.Event()

    @property
    def variant(self) -> str:
        """Return the execution variant identifier."""
        return "process_source"

    def __call__(self) -> TOut | None:
        """Produce an event or return None if nothing available."""
        raise NotImplementedError


class BaseThreadProcessor[TIn, TOut]:
    """Dedicated-thread processor. Subclass and override ``__call__``."""

    def __init__(self) -> None:
        """Initialize stop event for thread lifecycle management."""
        self.stop = threading.Event()

    @property
    def variant(self) -> str:
        """Return the execution variant identifier."""
        return "thread_processor"

    def __call__(self, event: TIn) -> TOut | None:
        """Transform an event or return None to filter it out."""
        raise NotImplementedError


class BaseThreadPoolProcessor[TIn, TOut]:
    """Thread-pool processor (N workers, shared queue).

    N worker threads call the same instance concurrently —
    ``__call__`` must be thread-safe.
    """

    def __init__(self, num_workers: int = 4) -> None:
        """Initialize with worker count and stop event."""
        self.stop = threading.Event()
        self.num_workers = num_workers

    @property
    def variant(self) -> str:
        """Return the execution variant identifier."""
        return "thread_pool_processor"

    def __call__(self, event: TIn) -> TOut | None:
        """Transform an event or return None to filter it out."""
        raise NotImplementedError


class BaseProcessPoolProcessor[TIn, TOut]:
    """Process-pool processor (N child processes via ProcessPoolExecutor).

    The node instance must be picklable — store config values in
    ``__init__``, lazy-init heavy objects on first ``__call__`` in
    each child process.  Events (arguments and return values) must
    also be picklable.  Does NOT preserve event ordering.
    """

    def __init__(self, num_workers: int = 8, *, use_shm: bool = True) -> None:
        """Initialize with worker count, stop event, and shared memory toggle."""
        self.stop = threading.Event()
        self.num_workers = num_workers
        self.use_shm = use_shm

    def __getstate__(self) -> dict[str, object]:
        """Exclude unpicklable stop event for process-pool serialization."""
        state = self.__dict__.copy()
        state.pop("stop", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore state and recreate stop event in child process."""
        self.__dict__.update(state)
        self.stop = threading.Event()

    @property
    def variant(self) -> str:
        """Return the execution variant identifier."""
        return "process_pool_processor"

    def __call__(self, event: TIn) -> TOut | None:
        """Transform an event or return None to filter it out."""
        raise NotImplementedError


class BaseThreadSink[TIn]:
    """Threaded sink with dedicated thread. Subclass and override ``__call__``."""

    def __init__(self) -> None:
        """Initialize stop event for thread lifecycle management."""
        self.stop = threading.Event()

    @property
    def variant(self) -> str:
        """Return the execution variant identifier."""
        return "thread_sink"

    def __call__(self, event: TIn) -> Any | None:  # noqa: ANN401
        """Consume an event and optionally return a metric event."""
        raise NotImplementedError


class BaseProcessSink[TIn]:
    """Process sink with dedicated OS process. Subclass and override ``__call__``."""

    def __init__(self) -> None:
        """Initialize stop event for process lifecycle management."""
        self.stop = threading.Event()

    @property
    def variant(self) -> str:
        """Return the execution variant identifier."""
        return "process_sink"

    def __call__(self, event: TIn) -> Any | None:  # noqa: ANN401
        """Consume an event and optionally return a metric event."""
        raise NotImplementedError
