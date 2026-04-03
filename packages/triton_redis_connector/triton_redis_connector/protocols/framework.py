"""Framework-internal protocols for variant detection and aggregation rules.

WrappedNode declares execution variant for fast-path detection.
AggregationRuleProtocol defines time-windowed event aggregation.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

TEvent_contra = TypeVar("TEvent_contra", contravariant=True)


@runtime_checkable
class WrappedNode(Protocol):
    """Framework wrapper that declares its execution variant explicitly.

    Used by PipelineRegistry._detect_variant as a fast path to avoid
    signature-based inference.
    """

    @property
    def variant(self) -> str:
        """Return the ExecutionVariant value for this wrapper."""
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

    def aggregate(self, count: int, value_sum: float) -> float:
        """Compute the aggregate value from event count and accumulated sum."""
        ...
