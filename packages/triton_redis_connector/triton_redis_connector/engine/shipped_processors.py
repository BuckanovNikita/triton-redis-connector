"""Reusable shipped Node classes for common pipeline patterns.

Provides ready-made building blocks using the ``PipelineEvent``
composition pattern: ClientPreprocessorNode, InferenceNode,
PostProcessorNode, LogErrorSinkNode.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger
from pydantic import BaseModel

from triton_redis_connector.engine.events import (
    Event,
    ImagePayload,
    PipelineEvent,
    TritonRequest,
)
from triton_redis_connector.errors.models import ConnectorError, InferenceError, PreprocessingError
from triton_redis_connector.inference.request import InferenceOutput
from triton_redis_connector.protocols import BaseThreadProcessor, BaseThreadSink

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class _DecoderProtocol(Protocol):
    def decode(self, raw: bytes) -> npt.NDArray[np.uint8]: ...


class _PreprocessorProtocol(Protocol):
    def preprocess(
        self,
        image: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float32], float, tuple[int, int]]: ...


class _PostprocessorCallable[TIn, TResult](Protocol):
    def __call__(self, event: TIn) -> TResult | None: ...


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

        self._grpcclient = grpcclient
        self._client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=grpc_url)

    def __call__(self, event: PipelineEvent[TCtx, TReq]) -> PipelineEvent[TCtx, TOut] | InferenceError | None:
        """Run inference and wrap output in PipelineEvent or return InferenceError."""
        req = event.payload
        start = time.monotonic()

        if req.raw_image is not None:
            input_tensor = self._grpcclient.InferInput(req.input_name, req.raw_image.shape, "FP32")
            input_tensor.set_data_from_numpy(req.raw_image)
        elif req.image_bytes is not None:
            import numpy as np  # noqa: PLC0415

            data = np.frombuffer(req.image_bytes, dtype=np.uint8)
            input_tensor = self._grpcclient.InferInput(req.input_name, [len(data)], "UINT8")
            input_tensor.set_data_from_numpy(data)
        else:
            return None

        outputs = [self._grpcclient.InferRequestedOutput(name) for name in req.output_names]

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
