"""Metric marker types for declarative event instrumentation.

Counter, Gauge, and GroupBy are dataclass markers attached to Event classes
or their fields via ``_metrics_`` and ``Annotated``. MetricRecorder is the
protocol implemented by MetricBuffer for recording metric data points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Counter:
    """Class-level metric: increment a named counter each time this event is produced.

    Declare on event classes via ``_metrics_`` class attribute::

        class DecodedImage(Event):
            _metrics_ = [Counter("decode_success")]
    """

    name: str


@dataclass(frozen=True, slots=True)
class Gauge:
    """Field-level metric: extract a field value and record it as a named metric.

    Declare on event fields via ``Annotated``::

        class DetectionResult(Event):
            inference_time_ms: Annotated[float, Gauge("inference_time_ms")]
    """

    name: str


@dataclass(frozen=True, slots=True)
class GroupBy:
    """Field-level metric: count events grouped by this field over a time window.

    Declare on event fields via ``Annotated``::

        class ImageEvent(Event):
            camera_no: Annotated[str, GroupBy("camera_events", window_s=60)]
    """

    name: str
    window_s: int = 60


@runtime_checkable
class MetricRecorder(Protocol):
    """Records metric data points. Implemented by MetricBuffer."""

    def record(self, event: object) -> None:
        """Append a metric event to the buffer."""
        ...
