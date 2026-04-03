"""Lifecycle and observer protocols.

LifecycleService for start/stop services, Startable/Stoppable for optional
lifecycle hooks, and ObserverProtocol for envelope observation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from triton_redis_connector.protocols.messaging import Envelope

TObserved = TypeVar("TObserved")


@runtime_checkable
class LifecycleService(Protocol):
    """Service with start/stop lifecycle. Implemented by MetricEmitter and TritonMetricsCollector."""

    def start(self) -> None:
        """Start the background service."""
        ...

    def stop(self) -> None:
        """Stop the background service."""
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
class Startable(Protocol):
    """Optional lifecycle hook called before first dispatch."""

    def start(self) -> None:
        """Initialize resources before processing begins."""
        ...


@runtime_checkable
class Stoppable(Protocol):
    """Optional lifecycle hook called during shutdown."""

    def close(self) -> None:
        """Release resources during shutdown."""
        ...
