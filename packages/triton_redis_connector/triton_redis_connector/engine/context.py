"""EventContext — per-event metadata accessible via contextvars.

User code calls current_context() to access request_id, source_stream,
and timing information without receiving the framework-internal Envelope.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True)
class EventContext:
    """Immutable snapshot of event metadata for the current processing scope.

    Set by the framework before dispatching an event to user code.
    Retrieved by user code via current_context().
    """

    message_id: str
    source_stream: str
    created_at: float


_context_var: ContextVar[EventContext] = ContextVar("event_context")


def current_context() -> EventContext:
    """Return the EventContext for the event currently being processed.

    Raises LookupError if called outside of an event processing scope.
    """
    return _context_var.get()


def set_context(ctx: EventContext) -> Token[EventContext]:
    """Set the EventContext for the current processing scope.

    Returns a Token that can be used to reset the context via
    reset_context(token).
    """
    return _context_var.set(ctx)


def reset_context(token: Token[EventContext]) -> None:
    """Reset the EventContext to the state before set_context was called."""
    _context_var.reset(token)
