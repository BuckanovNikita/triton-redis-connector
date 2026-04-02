"""Reusable shipped processors for common pipeline patterns.

Provides ready-made building blocks: DALI input wrapping, client-side
preprocessing, event serialization, and error logging.

Also contains new Node classes that work with the ``PipelineEvent``
composition pattern (ClientPreprocessorNode, InferenceNode,
PostProcessorNode, LogErrorSinkNode).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel

from triton_redis_connector.engine.context import current_context
from triton_redis_connector.engine.events import (
    Event,
    ImagePayload,
    PipelineEvent,
    TritonRequest,
)
from triton_redis_connector.errors.models import ConnectorError, InferenceError, PreprocessingError
from triton_redis_connector.inference.request import InferenceOutput
from triton_redis_connector.protocols import BaseThreadProcessor, BaseThreadSink, Envelope

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


class _PostprocessorCallable[TIn, TResult](Protocol):
    def __call__(self, event: TIn) -> TResult | None: ...


class DaliTritonRequestBuilder:
    """Wraps raw image bytes into TritonRequest for DALI ensemble pipelines.

    When Triton uses a DALI ensemble for server-side preprocessing,
    the client sends raw encoded image bytes rather than preprocessed tensors.

    Accepts events with ``image_bytes`` (already loaded) or ``image_path``
    (reads from filesystem).
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
        )


class ClientPreprocessor:
    """Full client-side preprocessing: read file, decode, letterbox, wrap.

    Reads an image file from the filesystem, decodes it using the provided
    decoder, applies letterbox preprocessing, and creates a TritonRequest
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

        Falls back to EventContext when available, or empty fields
        when running in a parallel thread without context.
        """
        payload = event.model_dump_json().encode()
        source_stream = ""
        message_id = ""
        original_fields: dict[str, str] = {}

        try:
            ctx = current_context()
            source_stream = ctx.source_stream
            message_id = ctx.message_id
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


# ---------------------------------------------------------------------------
# New Node classes using PipelineEvent composition pattern.
#
# These inherit from Base* classes for variant declaration and use
# the ``process()`` method pattern. Existing shipped processors above
# are kept for backward compatibility during migration.
# ---------------------------------------------------------------------------


class ClientPreprocessorNode[TCtx: Event, TPayload: ImagePayload](
    BaseThreadProcessor[PipelineEvent[TCtx, TPayload], PipelineEvent[TCtx, TritonRequest] | PreprocessingError],
):
    """PipelineEvent-aware client-side preprocessor.

    Reads image from ``ImagePayload`` (path or bytes), decodes, applies
    letterbox, and wraps as ``PipelineEvent[TCtx, TritonRequest]``.
    Returns ``PreprocessingError`` on failure instead of dropping.
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
        super().__init__()
        self._decoder = decoder
        self._preprocessor = preprocessor
        self._model_name = model_name
        self._input_name = input_name
        self._output_names = tuple(output_names)

    def __call__(
        self, event: PipelineEvent[TCtx, TPayload]
    ) -> PipelineEvent[TCtx, TritonRequest] | PreprocessingError | None:
        """Read image, preprocess, and wrap as PipelineEvent with TritonRequest payload."""
        payload = event.payload

        raw_bytes: bytes
        if payload.image_bytes is not None:
            raw_bytes = payload.image_bytes
        elif payload.image_path:
            try:
                raw_bytes = Path(payload.image_path).read_bytes()
            except (FileNotFoundError, PermissionError, OSError) as exc:
                return PreprocessingError(
                    message_id=event.context.fw.message_id,
                    decoder_type="filesystem",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    timestamp_ms=int(time.time() * 1000),
                )
        else:
            return None

        try:
            decoded = self._decoder.decode(raw_bytes)
        except ValueError as exc:
            return PreprocessingError(
                message_id=event.context.fw.message_id,
                decoder_type="decode",
                error_type="DecodeError",
                error_message=str(exc),
                timestamp_ms=int(time.time() * 1000),
            )

        tensor, _scale, _pad_offset = self._preprocessor.preprocess(decoded)
        triton_req = TritonRequest(
            model_name=self._model_name,
            raw_image=tensor,
            input_name=self._input_name,
            output_names=self._output_names,
        )
        return PipelineEvent(context=event.context, payload=triton_req)


class InferenceNode[TCtx: Event, TReq: TritonRequest, TOut: InferenceOutput](
    BaseThreadProcessor[PipelineEvent[TCtx, TReq], PipelineEvent[TCtx, TOut] | InferenceError],
):
    """PipelineEvent-aware Triton inference processor.

    Sends a ``TritonRequest`` to Triton and wraps the result in a
    ``PipelineEvent[TCtx, InferenceOutput]``. Returns ``InferenceError``
    on failure.
    """

    def __init__(self, grpc_url: str) -> None:  # noqa: D107
        super().__init__()
        import tritonclient.grpc as grpcclient  # noqa: PLC0415

        self._client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=grpc_url)

    def __call__(self, event: PipelineEvent[TCtx, TReq]) -> PipelineEvent[TCtx, TOut] | InferenceError | None:
        """Run inference and wrap output in PipelineEvent or return InferenceError."""
        import tritonclient.grpc as grpcclient  # noqa: PLC0415

        req = event.payload
        start = time.monotonic()

        if req.raw_image is not None:
            input_tensor = grpcclient.InferInput(req.input_name, req.raw_image.shape, "FP32")
            input_tensor.set_data_from_numpy(req.raw_image)
        elif req.image_bytes is not None:
            import numpy as np  # noqa: PLC0415

            data = np.frombuffer(req.image_bytes, dtype=np.uint8)
            input_tensor = grpcclient.InferInput(req.input_name, [len(data)], "UINT8")
            input_tensor.set_data_from_numpy(data)
        else:
            return None

        outputs = [grpcclient.InferRequestedOutput(name) for name in req.output_names]

        try:
            result = self._client.infer(
                model_name=req.model_name,
                inputs=[input_tensor],
                outputs=outputs,
                model_version=req.model_version or "",
            )
        except Exception as exc:  # noqa: BLE001
            return InferenceError(
                model_name=req.model_name,
                model_version=req.model_version,
                error_type=type(exc).__name__,
                error_message=str(exc),
                retry_count=0,
                timestamp_ms=int(time.time() * 1000),
            )

        if result is None:
            return InferenceError(
                model_name=req.model_name,
                model_version=req.model_version,
                error_type="NullResult",
                error_message=f"Triton returned None for model {req.model_name}",
                retry_count=0,
                timestamp_ms=int(time.time() * 1000),
            )

        import numpy as np  # noqa: PLC0415

        elapsed_ms = (time.monotonic() - start) * 1000.0
        output_arrays: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {
            name: np.asarray(result.as_numpy(name), dtype=np.float32) for name in req.output_names
        }

        inference_output = InferenceOutput(
            model_name=req.model_name,
            model_version=req.model_version,
            outputs=output_arrays,
            inference_time_ms=elapsed_ms,
        )
        return PipelineEvent(context=event.context, payload=inference_output)  # type: ignore[arg-type]


class PostProcessorNode[TCtx: Event, TIn: InferenceOutput, TResult: BaseModel](
    BaseThreadProcessor[PipelineEvent[TCtx, TIn], PipelineEvent[TCtx, TResult]],
):
    """PipelineEvent-aware postprocessor wrapper.

    Delegates to a user-supplied postprocessor callable that transforms
    ``InferenceOutput`` -> ``TResult``. Wraps the result back into
    ``PipelineEvent[TCtx, TResult]``.
    """

    def __init__(  # noqa: D107
        self,
        postprocessor: _PostprocessorCallable[TIn, TResult],
    ) -> None:
        super().__init__()
        self._postprocessor = postprocessor

    def __call__(self, event: PipelineEvent[TCtx, TIn]) -> PipelineEvent[TCtx, TResult] | None:
        """Run postprocessing and wrap result in PipelineEvent."""
        result = self._postprocessor(event.payload)
        if result is None:
            return None
        return PipelineEvent(context=event.context, payload=result)


class LogErrorSinkNode(
    BaseThreadSink[InferenceError | ConnectorError | PreprocessingError],
):
    """Logs all error event types at error level on a dedicated thread."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()

    def __call__(self, event: InferenceError | ConnectorError | PreprocessingError) -> None:
        """Log the error event fields at error level."""
        logger.error(
            "Pipeline error: {}: {}",
            type(event).__name__,
            event.error_message,
        )
