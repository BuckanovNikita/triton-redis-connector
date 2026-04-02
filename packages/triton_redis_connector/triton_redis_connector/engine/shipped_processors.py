"""Reusable shipped processors for common pipeline patterns.

Provides ready-made building blocks: DALI input wrapping, client-side
preprocessing, event serialization, and error logging.

All preprocessors support metadata bypass: if the input event carries a
``metadata`` dict, it is copied through to the ``TritonRequest`` without
interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel

from triton_redis_connector.engine.context import current_context
from triton_redis_connector.engine.events import TritonRequest
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
class _MetadataCarrier(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...


@runtime_checkable
class _DecoderProtocol(Protocol):
    def decode(self, raw: bytes) -> npt.NDArray[np.uint8]: ...


@runtime_checkable
class _PreprocessorProtocol(Protocol):
    def preprocess(
        self,
        image: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float32], float, tuple[int, int]]: ...


def _extract_metadata(event: object) -> dict[str, Any]:
    if isinstance(event, _MetadataCarrier):
        return dict(event.metadata)
    return {}


class DaliTritonRequestBuilder:
    """Wraps raw image bytes into TritonRequest for DALI ensemble pipelines.

    When Triton uses a DALI ensemble for server-side preprocessing,
    the client sends raw encoded image bytes rather than preprocessed tensors.

    Accepts events with ``image_bytes`` (already loaded) or ``image_path``
    (reads from filesystem). Bypasses metadata from the input event.
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
        self._output_names = tuple(output_names)

    def __call__(self, event: _ImageBytesEvent | _ImagePathEvent) -> TritonRequest | None:
        """Create TritonRequest with image_bytes for server-side decoding."""
        raw: bytes
        if isinstance(event, _ImageBytesEvent):
            raw = event.image_bytes
        else:
            try:
                raw = Path(event.image_path).read_bytes()
            except (FileNotFoundError, PermissionError, OSError):
                logger.exception("Failed to read image at {}", event.image_path)
                return None

        return TritonRequest(
            model_name=self._model_name,
            image_bytes=raw,
            input_name=self._input_name,
            output_names=self._output_names,
            metadata=_extract_metadata(event),
        )


class ClientPreprocessor:
    """Full client-side preprocessing: read file, decode, letterbox, wrap.

    Reads an image file from the filesystem, decodes it using the provided
    decoder, applies letterbox preprocessing, and creates a TritonRequest
    with the resulting tensor. Bypasses metadata from the input event.
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
        self._output_names = tuple(output_names)

    def __call__(self, event: _ImagePathEvent) -> TritonRequest | None:
        """Read image from filesystem, preprocess, and wrap as TritonRequest."""
        try:
            raw_bytes = Path(event.image_path).read_bytes()
        except (FileNotFoundError, PermissionError, OSError):
            logger.exception("Failed to read image at {}", event.image_path)
            return None

        try:
            decoded = self._decoder.decode(raw_bytes)
        except ValueError:
            logger.exception("Decode failed for {}", event.image_path)
            return None

        tensor, _scale, _pad_offset = self._preprocessor.preprocess(decoded)
        return TritonRequest(
            model_name=self._model_name,
            raw_image=tensor,
            input_name=self._input_name,
            output_names=self._output_names,
            metadata=_extract_metadata(event),
        )


class EventSerializer[T: BaseModel]:
    """Serializes a pydantic event into an Envelope[bytes] with original fields.

    Reads the current EventContext to preserve original_fields from
    the inbound message, ensuring they are forwarded through the pipeline.
    """

    def __init__(self, event_type: type[T] | None = None) -> None:  # noqa: D107
        if event_type is not None:
            self.__call_hints__: dict[str, type] = {
                "event": event_type,
                "return": Envelope[bytes],
            }

    def __call__(self, event: T) -> Envelope[bytes]:
        """Serialize event to JSON bytes and wrap in an Envelope.

        Populates ``original_fields`` from the event's metadata dict (if
        present) so downstream Redis sinks write them as top-level hash
        fields.  Falls back to EventContext when available, or empty fields
        when running in a parallel thread without context.
        """
        payload = event.model_dump_json().encode()
        metadata = _extract_metadata(event)
        original_fields = {k: str(v) for k, v in metadata.items()}
        source_stream = original_fields.pop("source_stream", "")
        message_id = original_fields.pop("message_id", "")

        try:
            ctx = current_context()
            source_stream = source_stream or ctx.source_stream
            message_id = message_id or ctx.message_id
            if not original_fields:
                original_fields = dict(ctx.original_fields)
        except LookupError:
            pass

        return Envelope(
            payload=payload,
            source_stream=source_stream,
            message_id=message_id,
            original_fields=original_fields,
        )


class LogErrorSink:
    """Logs pydantic error events at error level via loguru."""

    def __call__(self, event: BaseModel) -> None:
        """Log the error event fields at error level."""
        logger.error(
            "Pipeline error: {}",
            json.dumps(event.model_dump(), default=str),
        )
