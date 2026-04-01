"""Reusable shipped processors for common pipeline patterns.

Provides ready-made building blocks: envelope extraction, DALI input
wrapping, client-side preprocessing, event serialization, and error logging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ValidationError

from triton_redis_connector.engine.context import current_context
from triton_redis_connector.engine.events import InferenceInput
from triton_redis_connector.protocols import Envelope

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@runtime_checkable
class _ImageBytesEvent(Protocol):
    @property
    def image_bytes(self) -> bytes: ...


@runtime_checkable
class _ImagePathEvent(Protocol):
    @property
    def image_path(self) -> str: ...


@runtime_checkable
class _DecoderProtocol(Protocol):
    def decode(self, raw: bytes) -> npt.NDArray[np.uint8]: ...


@runtime_checkable
class _PreprocessorProtocol(Protocol):
    def preprocess(
        self,
        image: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float32], float, tuple[int, int]]: ...


class EnvelopeExtractor[T: BaseModel]:
    """Extracts a typed pydantic payload from an Envelope[bytes].

    Deserializes the envelope's bytes payload into a pydantic model
    of type T using model_validate_json.
    """

    def __init__(self, model_class: type[T]) -> None:  # noqa: D107
        self._model_class = model_class

    def __call__(self, event: Envelope[bytes]) -> T | None:
        """Deserialize envelope bytes payload into a pydantic model."""
        try:
            return self._model_class.model_validate_json(event.payload)
        except ValidationError:
            logger.exception(
                "Failed to deserialize {} from envelope {}",
                self._model_class.__name__,
                event.message_id,
            )
            return None


class DaliInferenceInputBuilder:
    """Wraps raw image bytes into InferenceInput for DALI ensemble pipelines.

    When Triton uses a DALI ensemble for server-side preprocessing,
    the client sends raw encoded image bytes rather than preprocessed tensors.
    """

    def __init__(  # noqa: D107
        self,
        model_name: str,
        *,
        input_name: str = "images",
        output_names: tuple[str, ...] = ("output0",),
    ) -> None:
        self._model_name = model_name
        self._input_name = input_name
        self._output_names = output_names

    def __call__(self, event: _ImageBytesEvent) -> InferenceInput:
        """Create InferenceInput with image_bytes for server-side decoding."""
        return InferenceInput(
            model_name=self._model_name,
            image_bytes=event.image_bytes,
            input_name=self._input_name,
            output_names=self._output_names,
        )


class ClientPreprocessor:
    """Full client-side preprocessing: read file, decode, letterbox, wrap.

    Reads an image file from the filesystem, decodes it using the provided
    decoder, applies letterbox preprocessing, and creates an InferenceInput
    with the resulting tensor.
    """

    def __init__(  # noqa: D107
        self,
        decoder: _DecoderProtocol,
        preprocessor: _PreprocessorProtocol,
        model_name: str,
        *,
        input_name: str = "images",
        output_names: tuple[str, ...] = ("output0",),
    ) -> None:
        self._decoder = decoder
        self._preprocessor = preprocessor
        self._model_name = model_name
        self._input_name = input_name
        self._output_names = output_names

    def __call__(self, event: _ImagePathEvent) -> InferenceInput:
        """Read image from filesystem, preprocess, and wrap as InferenceInput."""
        raw_bytes = Path(event.image_path).read_bytes()
        decoded = self._decoder.decode(raw_bytes)
        tensor, _scale, _pad_offset = self._preprocessor.preprocess(decoded)
        return InferenceInput(
            model_name=self._model_name,
            raw_image=tensor,
            input_name=self._input_name,
            output_names=self._output_names,
        )


class EventSerializer[T: BaseModel]:
    """Serializes a pydantic event into an Envelope[bytes] with original fields.

    Reads the current EventContext to preserve original_fields from
    the inbound message, ensuring they are forwarded through the pipeline.
    """

    def __call__(self, event: T) -> Envelope[bytes]:
        """Serialize event to JSON bytes and wrap in an Envelope."""
        payload = event.model_dump_json().encode()
        ctx = current_context()
        return Envelope(
            payload=payload,
            source_stream=ctx.source_stream,
            message_id=ctx.message_id,
            original_fields=dict(ctx.original_fields),
        )


class LogErrorSink:
    """Logs pydantic error events at error level via loguru."""

    def __call__(self, event: BaseModel) -> None:
        """Log the error event fields at error level."""
        logger.error(
            "Pipeline error: {}",
            json.dumps(event.model_dump(), default=str),
        )
