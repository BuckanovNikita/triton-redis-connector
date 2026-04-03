"""Typed error events that flow through the pipeline graph.

Errors are never raised during normal operation. Instead, processors and sources
return these typed error events, which the graph routes to error-handling sinks.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConnectorError(BaseModel):
    """Error originating from a Redis connector (source or sink).

    Covers connection failures, deserialization errors, and stream operation
    failures. Routed to ErrorHandlerSink via graph type matching.
    """

    model_config = ConfigDict(frozen=True)

    connector_name: str
    stream_name: str
    message_id: str | None
    error_type: str
    error_message: str
    timestamp_ms: int
