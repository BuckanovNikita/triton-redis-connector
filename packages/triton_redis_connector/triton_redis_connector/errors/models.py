"""Typed error events for Triton inference and preprocessing failures.

These errors flow through the pipeline graph as events, never raised.
All error types inherit from PipelineError so a single error sink
accepting PipelineError catches everything via inheritance routing.
"""

from typing import ClassVar

from typed_flow.errors.models import PipelineError
from typed_flow.protocols import Counter


class InferenceError(PipelineError):
    """Error from Triton inference processor.

    Emitted after all retry attempts are exhausted. Contains the model
    identity and retry count for diagnostics.
    """

    model_name: str
    model_version: str
    retry_count: int


class PreprocessingError(PipelineError):
    """Error from image decoding or preprocessing.

    Emitted when an image cannot be decoded or transformed to the
    expected tensor format.
    """

    _metrics_: ClassVar[list[Counter]] = [Counter("decode_failure")]

    message_id: str
    decoder_type: str
