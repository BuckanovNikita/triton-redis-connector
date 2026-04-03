"""Envelope: framework-internal event transport wrapper.

Envelope is generic over the payload type and carries timing, state tracking,
and source metadata through the processing graph. User code never creates or
receives Envelopes directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


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
