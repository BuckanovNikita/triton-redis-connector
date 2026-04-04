"""Reusable shipped Node classes for common pipeline patterns.

Provides ready-made building blocks for detection and classification
CV inference pipelines: preprocessors, inference nodes, postprocessors,
and error sinks.
"""

import time
import types as stdtypes
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import tritonclient.grpc as grpcclient
from loguru import logger
from pydantic import BaseModel
from typed_flow.errors.models import ConnectorError
from typed_flow.protocols import BaseProcessPoolProcessor, BaseThreadProcessor, BaseThreadSink
from typed_flow.protocols.metrics import TimeMetric

from triton_redis_connector.config.schema import DecoderType
from triton_redis_connector.engine.events import (
    ClassificationTritonRequest,
    DetectionTritonRequest,
    ImagePayload,
    InferenceTarget,
    TritonRequest,
)
from triton_redis_connector.errors.models import InferenceError, PreprocessingError
from triton_redis_connector.inference.request import (
    ClassificationInferenceOutput,
    DetectionInferenceOutput,
    InferenceOutput,
)
from triton_redis_connector.preprocessing.decoder import create_decoder
from triton_redis_connector.preprocessing.protocols import ImageDecoderProtocol, ImagePreprocessorProtocol


class _PostprocessorCallable[TIn, TResult](Protocol):
    def __call__(self, event: TIn) -> TResult | None: ...


def _read_and_decode(
    event: ImagePayload,
    decoder: ImageDecoderProtocol,
) -> npt.NDArray[np.uint8] | PreprocessingError | None:
    """Read image bytes from event and decode to BGR HWC uint8 array."""
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
        return decoder.decode(raw_bytes)
    except ValueError as exc:
        return PreprocessingError(
            message_id=event.flow_meta.message_id,
            decoder_type="decode",
            error_type="DecodeError",
            error_message=str(exc),
            timestamp_ms=TimeMetric.now(),
        )


def _resolve_model(
    event: ImagePayload,
    default_model: str,
) -> tuple[str, str]:
    """Resolve model name and version from event or defaults."""
    model = default_model
    version = ""
    if isinstance(event, InferenceTarget):
        if event.target_model:
            model = event.target_model
        if event.target_model_version:
            version = event.target_model_version
    return model, version


class ClientPreprocessorNode[TIn: ImagePayload](
    BaseProcessPoolProcessor[TIn, DetectionTritonRequest | PreprocessingError],
):
    """Client-side detection preprocessor: read, decode, letterbox, produce DetectionTritonRequest.

    Preserves preprocessing scale and pad_offset for postprocessing.
    Stores only picklable config — decoder and preprocessor are
    lazy-initialized on first call in each child process.
    """

    def __init__(  # noqa: PLR0913
        self,
        decoder_type: DecoderType,
        target_size: int | tuple[int, int],
        model_name: str,
        *,
        num_workers: int = 8,
        use_shm: bool = True,
        dtype: type[np.floating[Any]] = np.float32,
        input_name: str = "images",
        output_names: tuple[str, ...] = ("output0",),
    ) -> None:
        """Initialize preprocessor with picklable config for process-pool."""
        super().__init__(num_workers=num_workers, use_shm=use_shm)
        self._decoder_type = decoder_type
        self._target_size = target_size
        self._model_name = model_name
        self._dtype = dtype
        self._input_name = input_name
        self._output_names = tuple(output_names)
        self._decoder: ImageDecoderProtocol | None = None
        self._preprocessor: ImagePreprocessorProtocol | None = None

    def _ensure_initialized(self) -> tuple[ImageDecoderProtocol, ImagePreprocessorProtocol]:
        if self._decoder is None or self._preprocessor is None:
            from triton_shared_models.preprocessing import LetterBoxPreprocessor  # noqa: PLC0415

            self._decoder = create_decoder(self._decoder_type)
            self._preprocessor = LetterBoxPreprocessor(self._target_size, dtype=self._dtype)
        return self._decoder, self._preprocessor

    def __call__(self, event: TIn) -> DetectionTritonRequest | PreprocessingError | None:
        """Read image, preprocess, and produce DetectionTritonRequest with flow_meta."""
        decoder, preprocessor = self._ensure_initialized()

        decoded = _read_and_decode(event, decoder)
        if not isinstance(decoded, np.ndarray):
            return decoded

        tensor, scale, pad_offset = preprocessor.preprocess(decoded)
        model, version = _resolve_model(event, self._model_name)

        return DetectionTritonRequest(
            flow_meta=event.flow_meta,
            model_name=model,
            model_version=version,
            raw_image=tensor,
            input_name=self._input_name,
            output_names=self._output_names,
            preprocess_scale=scale,
            preprocess_pad=pad_offset,
        )


class ClassificationPreprocessorNode[TIn: ImagePayload](
    BaseProcessPoolProcessor[TIn, ClassificationTritonRequest | PreprocessingError],
):
    """Client-side classification preprocessor: read, decode, center-crop, produce ClassificationTritonRequest."""

    def __init__(  # noqa: PLR0913
        self,
        decoder_type: DecoderType,
        crop_size: int,
        model_name: str,
        *,
        num_workers: int = 8,
        use_shm: bool = True,
        input_name: str = "images",
        output_names: tuple[str, ...] = ("output0",),
    ) -> None:
        """Initialize classification preprocessor with picklable config."""
        super().__init__(num_workers=num_workers, use_shm=use_shm)
        self._decoder_type = decoder_type
        self._crop_size = crop_size
        self._model_name = model_name
        self._input_name = input_name
        self._output_names = tuple(output_names)
        self._decoder: ImageDecoderProtocol | None = None
        self._preprocessor: object | None = None

    def _ensure_initialized(self) -> tuple[ImageDecoderProtocol, object]:
        if self._decoder is None or self._preprocessor is None:
            from triton_shared_models.classification.preprocessing import CenterCropPreprocessor  # noqa: PLC0415

            self._decoder = create_decoder(self._decoder_type)
            self._preprocessor = CenterCropPreprocessor(crop_size=self._crop_size)
        return self._decoder, self._preprocessor

    def __call__(self, event: TIn) -> ClassificationTritonRequest | PreprocessingError | None:
        """Read image, preprocess with center crop, and produce ClassificationTritonRequest."""
        decoder, preprocessor = self._ensure_initialized()

        decoded = _read_and_decode(event, decoder)
        if not isinstance(decoded, np.ndarray):
            return decoded

        tensor = preprocessor.preprocess(decoded)  # type: ignore[attr-defined]
        model, version = _resolve_model(event, self._model_name)

        return ClassificationTritonRequest(
            flow_meta=event.flow_meta,
            model_name=model,
            model_version=version,
            raw_image=tensor,
            input_name=self._input_name,
            output_names=self._output_names,
        )


def _run_inference(
    client: grpcclient.InferenceServerClient,
    grpcclient_module: stdtypes.ModuleType,
    event: TritonRequest,
    output_cls: type[InferenceOutput],
) -> InferenceOutput | InferenceError | None:
    """Shared inference logic: build gRPC request, call Triton, wrap result."""
    start = time.monotonic()

    if event.raw_image is not None:
        dtype_str = "FP16" if event.raw_image.dtype == np.float16 else "FP32"
        input_tensor = grpcclient_module.InferInput(event.input_name, event.raw_image.shape, dtype_str)
        input_tensor.set_data_from_numpy(event.raw_image)
    elif event.image_bytes is not None:
        data = np.frombuffer(event.image_bytes, dtype=np.uint8)
        input_tensor = grpcclient_module.InferInput(event.input_name, [len(data)], "UINT8")
        input_tensor.set_data_from_numpy(data)
    else:
        return None

    outputs = [grpcclient_module.InferRequestedOutput(name) for name in event.output_names]

    try:
        result = client.infer(
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
    output_arrays: dict[str, npt.NDArray[np.float32]] = {
        name: np.asarray(result.as_numpy(name), dtype=np.float32) for name in event.output_names
    }

    return output_cls(
        flow_meta=event.flow_meta,
        model_name=event.model_name,
        model_version=event.model_version,
        outputs=output_arrays,
        inference_time_ms=TimeMetric.from_seconds(elapsed),
        preprocess_scale=event.preprocess_scale,
        preprocess_pad=event.preprocess_pad,
    )


class DetectionInferenceNode(
    BaseThreadProcessor[DetectionTritonRequest, DetectionInferenceOutput | InferenceError],
):
    """Triton inference for detection models. Preserves preprocessing metadata."""

    def __init__(self, grpc_url: str) -> None:
        """Initialize with a Triton gRPC endpoint URL."""
        super().__init__()
        self._grpcclient = grpcclient
        self._client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=grpc_url)

    def __call__(self, event: DetectionTritonRequest) -> DetectionInferenceOutput | InferenceError | None:
        """Run detection inference and return typed output."""
        return _run_inference(self._client, self._grpcclient, event, DetectionInferenceOutput)  # type: ignore[return-value]


class ClassificationInferenceNode(
    BaseThreadProcessor[ClassificationTritonRequest, ClassificationInferenceOutput | InferenceError],
):
    """Triton inference for classification models."""

    def __init__(self, grpc_url: str) -> None:
        """Initialize with a Triton gRPC endpoint URL."""
        super().__init__()
        self._grpcclient = grpcclient
        self._client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=grpc_url)

    def __call__(self, event: ClassificationTritonRequest) -> ClassificationInferenceOutput | InferenceError | None:
        """Run classification inference and return typed output."""
        return _run_inference(self._client, self._grpcclient, event, ClassificationInferenceOutput)  # type: ignore[return-value]


class DetectionPostProcessorNode[TResult: BaseModel](
    BaseThreadProcessor[DetectionInferenceOutput, TResult],
):
    """Postprocessor wrapper for detection inference outputs."""

    def __init__(self, postprocessor: _PostprocessorCallable[DetectionInferenceOutput, TResult]) -> None:
        """Initialize with a postprocessor callable."""
        super().__init__()
        self._postprocessor = postprocessor

    def __call__(self, event: DetectionInferenceOutput) -> TResult | None:
        """Run postprocessing and return result."""
        return self._postprocessor(event)


class ClassificationPostProcessorNode[TResult: BaseModel](
    BaseThreadProcessor[ClassificationInferenceOutput, TResult],
):
    """Postprocessor wrapper for classification inference outputs."""

    def __init__(self, postprocessor: _PostprocessorCallable[ClassificationInferenceOutput, TResult]) -> None:
        """Initialize with a postprocessor callable."""
        super().__init__()
        self._postprocessor = postprocessor

    def __call__(self, event: ClassificationInferenceOutput) -> TResult | None:
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
