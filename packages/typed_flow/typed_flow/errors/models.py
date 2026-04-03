"""Typed error events that flow through the pipeline graph.

Errors are never raised during normal operation. Instead, processors and sources
return these typed error events, which the graph routes to error-handling sinks.

PipelineError is the base for all error events. Any sink accepting
PipelineError will receive errors from every node in the graph.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from typed_flow.protocols.metrics import TimeMetric


class PipelineError(BaseModel):
    """Base error type for all pipeline errors.

    Any sink that accepts PipelineError will receive every error in the graph
    (connector errors, processing errors, etc.) via subtype routing.
    """

    model_config = ConfigDict(frozen=True)

    node_name: str
    error_type: str
    error_message: str
    timestamp_ms: TimeMetric


class ConnectorError(PipelineError):
    """Error originating from a Redis connector (source or sink).

    Covers connection failures, deserialization errors, and stream operation
    failures. Routed to any sink accepting PipelineError or ConnectorError.
    """

    connector_name: str
    stream_name: str
    message_id: str | None


class ProcessingError(PipelineError):
    """Error raised when a processor or sink node fails during __call__.

    Captures the exception type and message so error sinks can log or alert
    without the pipeline crashing.
    """

    event_type: str


def make_processing_error(
    node_name: str,
    exc: BaseException,
    event: object,
) -> ProcessingError:
    """Create a ProcessingError from a caught exception."""
    return ProcessingError(
        node_name=node_name,
        error_type=type(exc).__name__,
        error_message=str(exc),
        timestamp_ms=TimeMetric.now(),
        event_type=type(event).__name__,
    )
