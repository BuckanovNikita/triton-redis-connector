"""Typed error events for Triton inference and preprocessing failures.

These errors flow through the pipeline graph as events, never raised.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict
from typed_flow.protocols import Counter


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

    _metrics_: ClassVar[list[Counter]] = [Counter("decode_failure")]

    message_id: str
    decoder_type: str
    error_type: str
    error_message: str
    timestamp_ms: int
