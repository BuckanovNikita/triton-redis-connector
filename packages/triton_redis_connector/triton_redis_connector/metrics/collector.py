"""MetricEvent model for pipeline instrumentation.

All metric producers (transit tracking, Triton stats polling, error aggregation,
queue overflow tracking, business aggregation) emit MetricEvent instances that
flow into a MetricBuffer and are periodically flushed to Redis by MetricEmitter.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MetricEvent(BaseModel):
    """Single metric data point produced by any pipeline component.

    Sinks optionally return MetricEvent to report instrumentation data
    back into the graph (e.g., transit time, results written count).
    """

    model_config = ConfigDict(frozen=True)

    component_name: str
    measure: str
    value: float
    timestamp_ms: int
    tags: dict[str, str] = Field(default_factory=dict)
