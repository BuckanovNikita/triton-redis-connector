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


class InferenceError(BaseModel):
    """Error from Triton inference processor.

    Emitted after all retry attempts are exhausted. Contains the model
    identity and retry count for diagnostics.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str
    model_version: str
    error_type: str
    error_message: str
    retry_count: int
    timestamp_ms: int


class PreprocessingError(BaseModel):
    """Error from image decoding or preprocessing.

    Emitted when an image cannot be decoded or transformed to the
    expected tensor format.
    """

    model_config = ConfigDict(frozen=True)

    message_id: str
    decoder_type: str
    error_type: str
    error_message: str
    timestamp_ms: int
