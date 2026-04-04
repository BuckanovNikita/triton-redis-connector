"""Envelope, FrameworkMeta, and HasFlowMeta protocol.

Envelope is framework-internal event transport. FrameworkMeta carries
source metadata (stream, message_id, priority) through the pipeline.
HasFlowMeta is a runtime-checkable protocol for types that carry flow_meta.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class FrameworkMeta(BaseModel):
    """Framework-owned metadata carried through the pipeline on event.flow_meta.

    Set by source connectors from the Redis envelope. Stripped by
    sinks before writing to the output stream.
    """

    model_config = ConfigDict(frozen=True)

    source_stream: str = ""
    message_id: str = ""
    priority: int = 0


@runtime_checkable
class HasFlowMeta(Protocol):
    """Protocol for types that carry framework metadata.

    Sources produce events satisfying this protocol so the framework
    can track source_stream, message_id, and priority through the pipeline.
    """

    flow_meta: FrameworkMeta


class BaseEvent(BaseModel):
    """Base class for user events flowing through the pipeline.

    Provides ``flow_meta`` with a default so sources can populate it
    after deserialization. Inherit from this instead of plain BaseModel
    to get automatic framework metadata support.
    """

    model_config = ConfigDict(frozen=True)

    flow_meta: FrameworkMeta = FrameworkMeta()


@dataclass
class Envelope[T]:
    """Generic event wrapper with timing and state tracking.

    Framework-internal only. Carries payload through the processing graph
    along with metadata for observability (state transitions, timing).

    Supports ordering via __lt__ for use in PriorityQueue (newer envelopes
    have higher created_at values and sort as "greater", so oldest-first
    ordering is the default heap behavior).
    """

    payload: T
    created_at: float = field(default_factory=time.monotonic)
    states: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])
    source_stream: str = ""
    message_id: str = ""
    original_fields: dict[str, str] = field(default_factory=dict)

    def track_state(self, state: str) -> None:
        """Record a named state transition with current monotonic timestamp."""
        self.states.append((state, time.monotonic()))

    def transit_time_ms(self) -> float:
        """Return elapsed time in milliseconds since envelope creation."""
        return (time.monotonic() - self.created_at) * 1000.0

    def derive[U](self, new_payload: U) -> "Envelope[U]":
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

    def __lt__(self, other: "Envelope[Any]") -> bool:
        """Compare by created_at for PriorityQueue ordering (oldest first)."""
        return self.created_at < other.created_at
