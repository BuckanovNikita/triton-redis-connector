"""Reusable shipped Node classes for common pipeline patterns.

Provides ready-made building blocks for CV inference pipelines:
unified generic preprocessor, inference, and postprocessor nodes,
plus error sinks. All nodes are parameterized with marker types
for type-driven routing.
"""

import time
import types as stdtypes
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tritonclient.grpc as grpcclient
from loguru import logger
from pydantic import BaseModel
from triton_shared_models.classification.postprocessing import ClassificationPostprocessor
from triton_shared_models.detection.postprocessing import YoloPostprocessor
from typed_flow.errors.models import PipelineError
from typed_flow.protocols import BaseProcessPoolProcessor, BaseThreadProcessor, BaseThreadSink
from typed_flow.protocols.metrics import TimeMetric

from triton_redis_connector.config.schema import DecoderType, PreprocessStrategy
from triton_redis_connector.engine.events import (
    ClassificationResult,
    DetectionResult,
    ImagePayload,
    TritonRequest,
)
from triton_redis_connector.errors.models import InferenceError, PreprocessingError
from triton_redis_connector.inference.model_metadata import ModelMetadata, ModelMetadataCache
from triton_redis_connector.inference.request import (
    InferenceOutput,
)
from triton_redis_connector.preprocessing.decoder import create_decoder
from triton_redis_connector.preprocessing.protocols import ImageDecoderProtocol

type _PostprocessorCallable[TIn, TResult] = Callable[[TIn], TResult | None] | Callable[[TIn], TResult]


def _read_and_decode(
    event: ImagePayload,
    decoder: ImageDecoderProtocol,
    node_name: str,
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
                node_name=node_name,
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
            node_name=node_name,
            message_id=event.flow_meta.message_id,
            decoder_type="decode",
            error_type="DecodeError",
            error_message=str(exc),
            timestamp_ms=TimeMetric.now(),
        )


class PreprocessorNode[TIn: ImagePayload, TOut: TritonRequest](
    BaseProcessPoolProcessor[TIn, TOut | PreprocessingError],
):
    """Unified preprocessor: read, decode, preprocess, produce typed TritonRequest.

    Supports both letterbox (detection) and center-crop (classification)
    strategies. Auto-detects target_size and dtype from Triton model metadata.

    ``output_cls`` must be passed explicitly because ``__orig_class__``
    does not survive process-pool pickling.
    """

    def __init__(  # noqa: PLR0913
        self,
        decoder_type: DecoderType,
        strategy: PreprocessStrategy,
        grpc_url: str,
        output_cls: type[TOut],
        *,
        num_workers: int = 8,
        use_shm: bool = True,
    ) -> None:
        """Initialize preprocessor with picklable config for process-pool."""
        super().__init__(num_workers=num_workers, use_shm=use_shm)
        self._decoder_type = decoder_type
        self._strategy = strategy
        self._grpc_url = grpc_url
        self._output_cls = output_cls
        self._decoder: ImageDecoderProtocol | None = None
        self._preprocessor: object | None = None
        self._metadata_cache: ModelMetadataCache | None = None
        self._model_metadata: dict[str, ModelMetadata] = {}

    def _ensure_initialized(self, model_name: str) -> tuple[ImageDecoderProtocol, object, ModelMetadata]:
        if self._metadata_cache is None:
            client = grpcclient.InferenceServerClient(url=self._grpc_url)
            self._metadata_cache = ModelMetadataCache(client)

        if model_name not in self._model_metadata:
            self._model_metadata[model_name] = self._metadata_cache.get(model_name)

        meta = self._model_metadata[model_name]

        if self._decoder is None:
            self._decoder = create_decoder(self._decoder_type)

        if self._preprocessor is None:
            target_size = meta.target_size
            dtype: type[np.floating[Any]] = np.float16 if meta.input_dtype_str == "FP16" else np.float32
            if self._strategy == PreprocessStrategy.LETTERBOX:
                from triton_shared_models.preprocessing import LetterBoxPreprocessor  # noqa: PLC0415

                self._preprocessor = LetterBoxPreprocessor(target_size, dtype=dtype)
            else:
                from triton_shared_models.classification.preprocessing import CenterCropPreprocessor  # noqa: PLC0415

                self._preprocessor = CenterCropPreprocessor(crop_size=target_size[0])

        return self._decoder, self._preprocessor, meta

    def __call__(self, event: TIn) -> TOut | PreprocessingError | None:
        """Read image, preprocess, and produce typed TritonRequest with flow_meta."""
        decoder, preprocessor, _meta = self._ensure_initialized(event.model_name)

        decoded = _read_and_decode(event, decoder, f"preprocess:{event.model_name}")
        if not isinstance(decoded, np.ndarray):
            return decoded

        if self._strategy == PreprocessStrategy.LETTERBOX:
            tensor, scale, pad_offset = preprocessor.preprocess(decoded)  # type: ignore[attr-defined]
            return self._output_cls(
                flow_meta=event.flow_meta,
                model_name=event.model_name,
                raw_image=tensor,
                preprocess_scale=scale,
                preprocess_pad=pad_offset,
            )

        tensor = preprocessor.preprocess(decoded)  # type: ignore[attr-defined]
        return self._output_cls(
            flow_meta=event.flow_meta,
            model_name=event.model_name,
            raw_image=tensor,
        )


def _run_inference(  # noqa: PLR0913
    client: grpcclient.InferenceServerClient,
    grpcclient_module: stdtypes.ModuleType,
    event: TritonRequest,
    output_cls: type[InferenceOutput],
    metadata: ModelMetadata,
    node_name: str,
) -> InferenceOutput | InferenceError | None:
    """Shared inference logic: build gRPC request, call Triton, wrap result."""
    start = time.monotonic()

    if event.raw_image is not None:
        dtype_str = "FP16" if event.raw_image.dtype == np.float16 else "FP32"
        input_tensor = grpcclient_module.InferInput(metadata.input_name, event.raw_image.shape, dtype_str)
        input_tensor.set_data_from_numpy(event.raw_image)
    elif event.image_bytes is not None:
        data = np.frombuffer(event.image_bytes, dtype=np.uint8)
        input_tensor = grpcclient_module.InferInput(metadata.input_name, [len(data)], "UINT8")
        input_tensor.set_data_from_numpy(data)
    else:
        return None

    outputs = [grpcclient_module.InferRequestedOutput(name) for name in metadata.output_names]

    try:
        result = client.infer(
            model_name=event.model_name,
            inputs=[input_tensor],
            outputs=outputs,
            model_version=event.model_version or "",
        )
    except Exception as exc:  # noqa: BLE001
        return InferenceError(
            node_name=node_name,
            model_name=event.model_name,
            model_version=event.model_version,
            error_type=type(exc).__name__,
            error_message=str(exc),
            retry_count=0,
            timestamp_ms=TimeMetric.now(),
        )

    if result is None:
        return InferenceError(
            node_name=node_name,
            model_name=event.model_name,
            model_version=event.model_version,
            error_type="NullResult",
            error_message=f"Triton returned None for model {event.model_name}",
            retry_count=0,
            timestamp_ms=TimeMetric.now(),
        )

    elapsed = time.monotonic() - start
    output_arrays: dict[str, npt.NDArray[np.float32]] = {
        name: np.asarray(result.as_numpy(name), dtype=np.float32) for name in metadata.output_names
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


class InferenceNode[TIn: TritonRequest, TOut: InferenceOutput](
    BaseThreadProcessor[TIn, TOut | InferenceError],
):
    """Generic Triton inference node parameterized by request and output types.

    Resolves ``TOut`` via ``__orig_class__`` on first call (lazy, thread-safe).
    Reads tensor names from ``ModelMetadataCache``.
    """

    _output_cls: type[InferenceOutput] | None = None

    def __init__(self, grpc_url: str, metadata_cache: ModelMetadataCache) -> None:
        """Initialize with a Triton gRPC endpoint URL and metadata cache."""
        super().__init__()
        self._grpcclient = grpcclient
        self._client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=grpc_url)
        self._metadata_cache = metadata_cache

    def _resolve_output_cls(self) -> type[InferenceOutput]:
        cls = self._output_cls
        if cls is None:
            cls = typing.get_args(self.__orig_class__)[1]  # type: ignore[attr-defined]
            self._output_cls = cls
        return cls

    def __call__(self, event: TIn) -> TOut | InferenceError | None:
        """Run inference and return typed output."""
        output_cls = self._resolve_output_cls()
        meta = self._metadata_cache.get(event.model_name, event.model_version)
        node_name = f"inference:{event.model_name}"
        return _run_inference(self._client, self._grpcclient, event, output_cls, meta, node_name)  # type: ignore[return-value]


class PostProcessorNode[TIn: InferenceOutput, TResult: BaseModel](
    BaseThreadProcessor[TIn, TResult],
):
    """Generic postprocessor wrapper parameterized by input and result types."""

    def __init__(self, postprocessor: _PostprocessorCallable[TIn, TResult]) -> None:
        """Initialize with a postprocessor callable."""
        super().__init__()
        self._postprocessor = postprocessor

    def __call__(self, event: TIn) -> TResult | None:
        """Run postprocessing and return result."""
        return self._postprocessor(event)


class DetectionPostprocessorCallable[TResult: DetectionResult]:
    """Composable detection postprocessor callable for PostProcessorNode.

    Wraps a YoloPostprocessor and a result class to produce typed
    DetectionResult events from raw inference outputs.
    """

    def __init__(self, result_cls: type[TResult], postprocessor: YoloPostprocessor | None = None) -> None:
        """Initialize with result class and optional custom postprocessor."""
        self._result_cls = result_cls
        self._postprocessor = postprocessor or YoloPostprocessor()

    def __call__(self, event: InferenceOutput) -> TResult:
        """Run YOLO postprocessing and wrap into typed result."""
        detections = self._postprocessor.postprocess(
            event.outputs,
            event.preprocess_scale,
            event.preprocess_pad,
        )
        return self._result_cls(
            flow_meta=event.flow_meta,
            detections=tuple(detections),
            model_name=event.model_name,
            inference_time_ms=event.inference_time_ms,
        )


class ClassificationPostprocessorCallable[TResult: ClassificationResult]:
    """Composable classification postprocessor callable for PostProcessorNode.

    Wraps a ClassificationPostprocessor and a result class to produce typed
    ClassificationResult events from raw inference outputs.
    """

    def __init__(self, result_cls: type[TResult], postprocessor: ClassificationPostprocessor | None = None) -> None:
        """Initialize with result class and optional custom postprocessor."""
        self._result_cls = result_cls
        self._postprocessor = postprocessor or ClassificationPostprocessor(top_k=5)

    def __call__(self, event: InferenceOutput) -> TResult:
        """Run classification postprocessing and wrap into typed result."""
        predictions = self._postprocessor.postprocess(event.outputs)
        return self._result_cls(
            flow_meta=event.flow_meta,
            predictions=tuple(predictions),
            model_name=event.model_name,
            inference_time_ms=event.inference_time_ms,
        )


class LogErrorSinkNode(
    BaseThreadSink[PipelineError],
):
    """Logs all pipeline error types via inheritance routing.

    Accepts PipelineError base class — catches all error subtypes
    (InferenceError, PreprocessingError, ConnectorError, ProcessingError)
    automatically through the framework's subclass routing.
    """

    def __init__(self) -> None:
        """Initialize the error logging sink."""
        super().__init__()

    def __call__(self, event: PipelineError) -> None:
        """Log the error event fields at error level."""
        if event.node_name:
            logger.error("[{}] {}: {}", event.node_name, event.error_type, event.error_message)
        else:
            logger.error("{}: {}", event.error_type, event.error_message)
