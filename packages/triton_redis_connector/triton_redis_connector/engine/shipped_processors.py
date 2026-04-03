"""Reusable shipped Node classes for common pipeline patterns.

Provides ready-made building blocks: ClientPreprocessorNode, InferenceNode,
PostProcessorNode, LogErrorSinkNode. Each processor is a simple Tin -> Tout
transform, with flow_meta threaded through pipeline types.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import tritonclient.grpc as grpcclient
from loguru import logger
from pydantic import BaseModel
from typed_flow.errors.models import ConnectorError
from typed_flow.protocols import BaseThreadProcessor, BaseThreadSink
from typed_flow.protocols.metrics import TimeMetric

from triton_redis_connector.engine.events import (
    ImagePayload,
    InferenceTarget,
    TritonRequest,
)
from triton_redis_connector.errors.models import InferenceError, PreprocessingError
from triton_redis_connector.inference.request import InferenceOutput


class _DecoderProtocol(Protocol):
    def decode(self, raw: bytes) -> npt.NDArray[np.uint8]: ...


class _PreprocessorProtocol(Protocol):
    def preprocess(
        self,
        image: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float32], float, tuple[int, int]]: ...


class _PostprocessorCallable[TIn, TResult](Protocol):
    def __call__(self, event: TIn) -> TResult | None: ...


class ClientPreprocessorNode[TIn: ImagePayload](
    BaseThreadProcessor[TIn, TritonRequest | PreprocessingError],
):
    """Client-side preprocessor that reads image, decodes, and produces TritonRequest.

    Input must satisfy HasFlowMeta (for error reporting) and ImagePayload
    (for image data). Returns PreprocessingError on failure.
    """

    def __init__(
        self,
        decoder: _DecoderProtocol,
        preprocessor: _PreprocessorProtocol,
        model_name: str,
        *,
        input_name: str = "images",
        output_names: tuple[str, ...] = ("output0",),
    ) -> None:
        """Initialize preprocessor with decoder, letterbox, and model config."""
        super().__init__()
        self._decoder = decoder
        self._preprocessor = preprocessor
        self._model_name = model_name
        self._input_name = input_name
        self._output_names = tuple(output_names)

    def __call__(self, event: TIn) -> TritonRequest | PreprocessingError | None:
        """Read image, preprocess, and produce TritonRequest with flow_meta."""
        raw_bytes: bytes
        if event.image_bytes is not None:
            raw_bytes = event.image_bytes
        elif event.image_path:
            try:
                raw_bytes = Path(event.image_path).read_bytes()
            except (FileNotFoundError, PermissionError, OSError) as exc:
                return PreprocessingError(
                    message_id=event.flow_meta.message_id,
                    decoder_type="filesystem",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    timestamp_ms=TimeMetric.now(),
                )
        else:
            return None

        try:
            decoded = self._decoder.decode(raw_bytes)
        except ValueError as exc:
            return PreprocessingError(
                message_id=event.flow_meta.message_id,
                decoder_type="decode",
                error_type="DecodeError",
                error_message=str(exc),
                timestamp_ms=TimeMetric.now(),
            )

        tensor, _scale, _pad_offset = self._preprocessor.preprocess(decoded)

        model = self._model_name
        version = ""
        if isinstance(event, InferenceTarget):
            if event.target_model:
                model = event.target_model
            if event.target_model_version:
                version = event.target_model_version

        return TritonRequest(
            flow_meta=event.flow_meta,
            model_name=model,
            model_version=version,
            raw_image=tensor,
            input_name=self._input_name,
            output_names=self._output_names,
        )


class InferenceNode(
    BaseThreadProcessor[TritonRequest, InferenceOutput | InferenceError],
):
    """Triton inference processor.

    Sends a TritonRequest to Triton and returns InferenceOutput with
    flow_meta preserved. Returns InferenceError on failure.
    """

    def __init__(self, grpc_url: str) -> None:
        """Initialize with a Triton gRPC endpoint URL."""
        super().__init__()
        self._grpcclient = grpcclient
        self._client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=grpc_url)

    def __call__(self, event: TritonRequest) -> InferenceOutput | InferenceError | None:
        """Run inference and return InferenceOutput or InferenceError."""
        start = time.monotonic()

        if event.raw_image is not None:
            input_tensor = self._grpcclient.InferInput(event.input_name, event.raw_image.shape, "FP32")
            input_tensor.set_data_from_numpy(event.raw_image)
        elif event.image_bytes is not None:
            data = np.frombuffer(event.image_bytes, dtype=np.uint8)
            input_tensor = self._grpcclient.InferInput(event.input_name, [len(data)], "UINT8")
            input_tensor.set_data_from_numpy(data)
        else:
            return None

        outputs = [self._grpcclient.InferRequestedOutput(name) for name in event.output_names]

        try:
            result = self._client.infer(
                model_name=event.model_name,
                inputs=[input_tensor],
                outputs=outputs,
                model_version=event.model_version or "",
            )
        except Exception as exc:  # noqa: BLE001
            return InferenceError(
                model_name=event.model_name,
                model_version=event.model_version,
                error_type=type(exc).__name__,
                error_message=str(exc),
                retry_count=0,
                timestamp_ms=TimeMetric.now(),
            )

        if result is None:
            return InferenceError(
                model_name=event.model_name,
                model_version=event.model_version,
                error_type="NullResult",
                error_message=f"Triton returned None for model {event.model_name}",
                retry_count=0,
                timestamp_ms=TimeMetric.now(),
            )

        elapsed = time.monotonic() - start
        output_arrays: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {
            name: np.asarray(result.as_numpy(name), dtype=np.float32) for name in event.output_names
        }

        return InferenceOutput(
            flow_meta=event.flow_meta,
            model_name=event.model_name,
            model_version=event.model_version,
            outputs=output_arrays,
            inference_time_ms=TimeMetric.from_seconds(elapsed),
        )


class PostProcessorNode[TIn: InferenceOutput, TResult: BaseModel](
    BaseThreadProcessor[TIn, TResult],
):
    """Postprocessor wrapper that delegates to a user-supplied callable.

    Transforms InferenceOutput -> TResult via the postprocessor callable.
    """

    def __init__(
        self,
        postprocessor: _PostprocessorCallable[TIn, TResult],
    ) -> None:
        """Initialize with a postprocessor callable."""
        super().__init__()
        self._postprocessor = postprocessor

    def __call__(self, event: TIn) -> TResult | None:
        """Run postprocessing and return result."""
        return self._postprocessor(event)


class LogErrorSinkNode(
    BaseThreadSink[InferenceError | ConnectorError | PreprocessingError],
):
    """Logs all error event types at error level on a dedicated thread."""

    def __init__(self) -> None:
        """Initialize the error logging sink."""
        super().__init__()

    def __call__(self, event: InferenceError | ConnectorError | PreprocessingError) -> None:
        """Log the error event fields at error level."""
        logger.error(
            "Pipeline error: {}: {}",
            type(event).__name__,
            event.error_message,
        )
