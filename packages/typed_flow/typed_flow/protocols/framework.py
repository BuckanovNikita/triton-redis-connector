"""Framework-internal protocols for variant detection and aggregation rules.

WrappedNode declares execution variant for fast-path detection.
AggregationRuleProtocol defines time-windowed event aggregation.
"""

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

TEvent_contra = TypeVar("TEvent_contra", contravariant=True)


@dataclass(frozen=True, slots=True)
class BucketStats:
    """Aggregated statistics for a single key within a time-window bucket."""

    count: int
    value_sum: float
    value_min: float
    value_max: float


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

    def aggregate(self, stats: BucketStats) -> float:
        """Compute the aggregate value from bucket statistics."""
        ...
