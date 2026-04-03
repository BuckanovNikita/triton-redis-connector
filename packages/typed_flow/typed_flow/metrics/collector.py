"""MetricEvent model for pipeline instrumentation.

All metric producers (transit tracking, external stats polling, error aggregation,
queue overflow tracking, business aggregation) emit MetricEvent instances that
flow into a MetricBuffer and are periodically flushed to Redis by MetricEmitter.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MetricEvent(BaseModel):
    """Single metric data point produced by any pipeline component.

    Contains only the essential triple: component, measure, value.
    Timestamp is provided by the Redis Stream entry ID at write time.
    """

    model_config = ConfigDict(frozen=True)

    component_name: str
    measure: str
    value: float
