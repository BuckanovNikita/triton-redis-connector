"""Metric marker types for declarative event instrumentation.

Counter, Gauge, and GroupBy are dataclass markers attached to Event classes
or their fields via ``_metrics_`` and ``Annotated``. MetricRecorder is the
protocol implemented by MetricBuffer for recording metric data points.
"""

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Self, runtime_checkable

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

_METRIC_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_metric_name(name: str) -> None:
    if not _METRIC_NAME_RE.match(name):
        msg = f"Invalid metric name {name!r}: must match [a-z][a-z0-9_]* (lowercase latin, digits, underscores)"
        raise ValueError(msg)


class TimeMetric(int):
    """Integer milliseconds type for Pydantic models.

    Constructor treats int/float as raw milliseconds (truncated to int).
    Use classmethods for conversions from other units.

    Usage::

        class MyEvent(BaseModel):
            inference_time_ms: Annotated[TimeMetric, Gauge("inference_time_ms")]
            timestamp_ms: TimeMetric


        TimeMetric.now()  # current wall-clock ms
        TimeMetric(1500)  # 1500ms
        TimeMetric.from_seconds(1.5)  # 1500ms
        TimeMetric.from_datetime(dt)  # datetime -> ms
    """

    def __new__(cls, value: float = 0) -> Self:  # noqa: D102
        return super().__new__(cls, int(value))

    @classmethod
    def now(cls) -> Self:
        """Return current wall-clock time as integer milliseconds."""
        return cls(int(time.time() * 1000))

    @classmethod
    def from_seconds(cls, seconds: float) -> Self:
        """Convert seconds (float) to integer milliseconds."""
        return cls(int(seconds * 1000))

    @classmethod
    def from_datetime(cls, dt: datetime) -> Self:
        """Convert a datetime to integer milliseconds since epoch."""
        return cls(int(dt.timestamp() * 1000))

    @classmethod
    def __get_pydantic_core_schema__(  # noqa: D105
        cls,
        source_type: type,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=core_schema.plain_serializer_function_ser_schema(int),
        )

    @classmethod
    def _pydantic_validate(cls, value: object) -> Self:
        if isinstance(value, datetime):
            return cls.from_datetime(value)
        if isinstance(value, (int, float)):
            return cls(int(value))
        msg = f"Cannot convert {type(value).__name__} to TimeMetric"
        raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class Counter:
    """Class-level metric: increment a named counter each time this event is produced.

    Declare on event classes via ``_metrics_`` class attribute::

        class DecodedImage(Event):
            _metrics_ = [Counter("decode_success")]
    """

    name: str

    def __post_init__(self) -> None:  # noqa: D105
        _validate_metric_name(self.name)


@dataclass(frozen=True, slots=True)
class Gauge:
    """Field-level metric: extract a field value and record it as a named metric.

    Declare on event fields via ``Annotated``::

        class ProcessingResult(Event):
            processing_time_ms: Annotated[float, Gauge("processing_time_ms")]
    """

    name: str

    def __post_init__(self) -> None:  # noqa: D105
        _validate_metric_name(self.name)


@dataclass(frozen=True, slots=True)
class GroupBy:
    """Field-level metric: count events grouped by this field over a time window.

    Declare on event fields via ``Annotated``::

        class ImageEvent(Event):
            camera_no: Annotated[str, GroupBy("camera_events", window_s=60)]
    """

    name: str
    window_s: int = 60

    def __post_init__(self) -> None:  # noqa: D105
        _validate_metric_name(self.name)


@runtime_checkable
class MetricRecorder(Protocol):
    """Records metric data points. Implemented by MetricBuffer."""

    def record(self, event: object) -> None:
        """Append a metric event to the buffer."""
        ...
