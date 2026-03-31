"""Protocol definitions, Envelope wrapper, and all 9 execution variants.

This is the leaf layer in the import-linter hierarchy. Every other module
in this package may import from protocols.py, but protocols.py has zero
internal dependencies.

All protocols are runtime_checkable for isinstance verification during
graph construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    import threading


T = TypeVar("T")
U = TypeVar("U")
TOut_co = TypeVar("TOut_co", covariant=True)
TIn_contra = TypeVar("TIn_contra", contravariant=True)
TShared = TypeVar("TShared")
TProcessor_co = TypeVar("TProcessor_co", covariant=True)
TObserved = TypeVar("TObserved")
TEvent_contra = TypeVar("TEvent_contra", contravariant=True)


@dataclass
class Envelope(Generic[T]):  # noqa: UP046
    """Generic event wrapper with timing and state tracking.

    Framework-internal only. Carries payload through the processing graph
    along with metadata for observability (state transitions, timing).

    Supports ordering via __lt__ for use in PriorityQueue (newer envelopes
    have higher created_at values and sort as "greater", so oldest-first
    ordering is the default heap behavior).
    """

    payload: T
    created_at: float = field(default_factory=time.monotonic)
    states: list[tuple[str, float]] = field(
        default_factory=lambda: list[tuple[str, float]](),  # noqa: PLW0108
    )
    source_stream: str = ""
    message_id: str = ""
    original_fields: dict[str, str] = field(default_factory=dict)

    def track_state(self, state: str) -> None:
        """Record a named state transition with current monotonic timestamp."""
        self.states.append((state, time.monotonic()))

    def transit_time_ms(self) -> float:
        """Return elapsed time in milliseconds since envelope creation."""
        return (time.monotonic() - self.created_at) * 1000.0

    def derive(self, new_payload: U) -> Envelope[U]:
        """Create a new Envelope carrying a different payload type.

        Preserves source_stream, message_id, created_at, and copies the
        existing state history. The derived envelope continues the same
        logical event's lifecycle.
        """
        return Envelope(
            payload=new_payload,
            created_at=self.created_at,
            states=list(self.states),
            source_stream=self.source_stream,
            message_id=self.message_id,
            original_fields=dict(self.original_fields),
        )

    def __lt__(self, other: Envelope[Any]) -> bool:
        """Compare by created_at for PriorityQueue ordering (oldest first)."""
        return self.created_at < other.created_at


@runtime_checkable
class Source(Protocol[TOut_co]):
    """Inline source. Runs in main loop, polled by framework.

    Framework calls __call__() repeatedly. Return an event when one is
    ready, None when nothing available. Must not block.
    """

    def __call__(self) -> TOut_co | None:
        """Produce an event or return None if nothing available."""
        ...


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
class Processor(Protocol[TIn_contra, TOut_co]):
    """Inline processor. Runs in main loop, framework dispatches events.

    Return None to filter (drop this event).
    """

    def __call__(self, event: TIn_contra) -> TOut_co | None:
        """Transform an event or return None to filter it out."""
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
class Sink(Protocol[TIn_contra]):
    """Inline sink. Runs in main loop.

    Terminal node with side effects. Can optionally return metric events.
    """

    def __call__(self, event: TIn_contra) -> Any | None:  # noqa: ANN401
        """Consume an event and optionally return a metric event."""
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
class ProcessorFactory(Protocol[TShared, TProcessor_co]):
    """Factory for creating processor instances with shared state.

    Framework calls create_shared() once, then create_processor(shared)
    N times (one per worker). User is responsible for thread/process
    safety of the shared context.
    """

    def create_shared(self) -> TShared:
        """Create the shared context object (called once)."""
        ...

    def create_processor(self, shared: TShared) -> TProcessor_co:
        """Create a processor instance with access to shared context."""
        ...


@runtime_checkable
class ObserverProtocol(Protocol[TObserved]):
    """Observes events without producing output (metrics, logging).

    Receives the full Envelope for access to timing and state data.
    """

    def observe(self, envelope: Envelope[TObserved]) -> None:
        """Observe an envelope without modifying or consuming it."""
        ...


@runtime_checkable
class AggregationRuleProtocol(Protocol[TEvent_contra]):
    """Extracts grouping key from events for time-windowed aggregation.

    Used by TimeWindowAggregator to bucket events and compute metrics
    over sliding windows.
    """

    @property
    def metric_name(self) -> str:
        """Name of the metric this rule produces."""
        ...

    @property
    def window_seconds(self) -> int:
        """Duration of the aggregation window in seconds."""
        ...

    def extract_key(self, event: TEvent_contra) -> str:
        """Extract the grouping key from an event."""
        ...

    def aggregate(self, count: int) -> float:
        """Compute the aggregate value from an event count."""
        ...
