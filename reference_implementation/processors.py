"""Pipeline processors for the CV inference pipeline.

Router, per-model preprocessors, inference wrappers, and postprocessors.
Minimizes code by inheriting from shipped processors where possible.
"""

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tritonclient.grpc as grpcclient
from loguru import logger
from triton_model_tools.inference.classification_postprocessing import ClassificationPostprocessor
from triton_model_tools.inference.postprocessing import YoloPostprocessor
from triton_redis_connector.config.schema import DecoderType
from triton_redis_connector.engine.events import (
    ClassificationResult,
    DetectionResult,
    ImagePayload,
    TritonRequest,
)
from triton_redis_connector.errors.models import InferenceError, PreprocessingError
from triton_redis_connector.inference.request import InferenceOutput
from triton_redis_connector.preprocessing.decoder import create_decoder
from triton_redis_connector.preprocessing.protocols import ImageDecoderProtocol
from typed_flow.protocols import BaseProcessPoolProcessor, BaseThreadProcessor
from typed_flow.protocols.metrics import TimeMetric

from reference_implementation.models import (
    CameraEvent,
    ClassModel3Event,
    DetModel1Event,
    DetModel2Event,
    DetModel4Event,
    Model1InfOut,
    Model1Request,
    Model2InfOut,
    Model2Request,
    Model3InfOut,
    Model3Request,
    Model4InfOut,
    Model4Request,
    RouteEvent,
)

CAM_MODEL1_AND_3 = {1, 2}
CAM_MODEL2 = {3, 4}
CAM_MODEL4 = {5, 6}


class CameraRouterProcessor(
    BaseThreadProcessor[CameraEvent, RouteEvent],
):
    """Routes camera events to model-specific event types by cam_id."""

    def __call__(  # type: ignore[override]
        self,
        event: CameraEvent,
    ) -> list[RouteEvent] | RouteEvent | None:
        """Route by cam_id: 1-2 to model1+3, 3-4 to model2, 5-6 to model4."""
        if event.cam_id in CAM_MODEL1_AND_3:
            return [
                DetModel1Event(
                    flow_meta=event.flow_meta,
                    image_path=event.image_path,
                    image_bytes=event.image_bytes,
                    cam_id=event.cam_id,
                ),
                ClassModel3Event(
                    flow_meta=event.flow_meta,
                    image_path=event.image_path,
                    image_bytes=event.image_bytes,
                    cam_id=event.cam_id,
                ),
            ]
        if event.cam_id in CAM_MODEL2:
            return DetModel2Event(
                flow_meta=event.flow_meta,
                image_path=event.image_path,
                image_bytes=event.image_bytes,
                cam_id=event.cam_id,
            )
        if event.cam_id in CAM_MODEL4:
            return DetModel4Event(
                flow_meta=event.flow_meta,
                image_path=event.image_path,
                image_bytes=event.image_bytes,
                cam_id=event.cam_id,
            )
        logger.warning("Unknown cam_id={}, dropping event", event.cam_id)
        return None


def _read_image_bytes(event: ImagePayload) -> bytes | PreprocessingError | None:
    """Read raw image bytes from event (path or in-memory)."""
    if event.image_bytes is not None:
        return event.image_bytes
    if event.image_path:
        try:
            return Path(event.image_path).read_bytes()
        except (FileNotFoundError, PermissionError, OSError) as exc:
            return PreprocessingError(
                message_id=event.flow_meta.message_id,
                decoder_type="filesystem",
                error_type=type(exc).__name__,
                error_message=str(exc),
                timestamp_ms=TimeMetric.now(),
            )
    return None


def _decode_image(
    decoder: ImageDecoderProtocol,
    raw_bytes: bytes,
    message_id: str,
) -> npt.NDArray[np.uint8] | PreprocessingError:
    """Decode raw bytes into BGR HWC uint8 array."""
    try:
        return decoder.decode(raw_bytes)
    except ValueError as exc:
        return PreprocessingError(
            message_id=message_id,
            decoder_type="decode",
            error_type="DecodeError",
            error_message=str(exc),
            timestamp_ms=TimeMetric.now(),
        )


def _preprocess_detection(
    event: ImagePayload,
    model_name: str,
    decoder: ImageDecoderProtocol,
    preprocessor: object,
    request_cls: type,
) -> object:
    """Shared detection preprocessing: read, decode, letterbox, build typed request."""
    raw = _read_image_bytes(event)
    if raw is None or isinstance(raw, PreprocessingError):
        return raw
    decoded = _decode_image(decoder, raw, event.flow_meta.message_id)
    if isinstance(decoded, PreprocessingError):
        return decoded

    tensor, scale, pad_offset = preprocessor.preprocess(decoded)  # type: ignore[attr-defined]
    return request_cls(
        flow_meta=event.flow_meta,
        model_name=model_name,
        raw_image=tensor,
        preprocess_scale=scale,
        preprocess_pad=pad_offset,
    )


class Model1Preprocessor(
    BaseProcessPoolProcessor[DetModel1Event, Model1Request | PreprocessingError],
):
    """640x640 TurboJPEG LetterBox preprocessor without shared memory."""

    def __init__(self, model_name: str, *, num_workers: int = 8) -> None:  # noqa: D107
        super().__init__(num_workers=num_workers, use_shm=False)
        self._model_name = model_name
        self._decoder: ImageDecoderProtocol | None = None
        self._preprocessor: object | None = None

    def _ensure_initialized(self) -> None:
        if self._decoder is None:
            from triton_shared_models.preprocessing import LetterBoxPreprocessor  # noqa: PLC0415

            self._decoder = create_decoder(DecoderType.TURBOJPEG)
            self._preprocessor = LetterBoxPreprocessor(target_size=640)

    def __call__(self, event: DetModel1Event) -> Model1Request | PreprocessingError | None:  # noqa: D102
        self._ensure_initialized()
        return _preprocess_detection(event, self._model_name, self._decoder, self._preprocessor, Model1Request)  # type: ignore[arg-type,return-value]


class Model2Preprocessor(
    BaseProcessPoolProcessor[DetModel2Event, Model2Request | PreprocessingError],
):
    """608x960 OpenCV FP16 LetterBox preprocessor without shared memory."""

    def __init__(self, model_name: str, *, num_workers: int = 8) -> None:  # noqa: D107
        super().__init__(num_workers=num_workers, use_shm=False)
        self._model_name = model_name
        self._decoder: ImageDecoderProtocol | None = None
        self._preprocessor: object | None = None

    def _ensure_initialized(self) -> None:
        if self._decoder is None:
            from triton_shared_models.preprocessing import LetterBoxPreprocessor  # noqa: PLC0415

            self._decoder = create_decoder(DecoderType.OPENCV)
            self._preprocessor = LetterBoxPreprocessor(target_size=(608, 960), dtype=np.float16)

    def __call__(self, event: DetModel2Event) -> Model2Request | PreprocessingError | None:  # noqa: D102
        self._ensure_initialized()
        return _preprocess_detection(event, self._model_name, self._decoder, self._preprocessor, Model2Request)  # type: ignore[arg-type,return-value]


class Model3Preprocessor(
    BaseProcessPoolProcessor[ClassModel3Event, Model3Request | PreprocessingError],
):
    """640x640 Classification CenterCrop preprocessor without shared memory."""

    def __init__(self, model_name: str, *, num_workers: int = 8) -> None:  # noqa: D107
        super().__init__(num_workers=num_workers, use_shm=False)
        self._model_name = model_name
        self._decoder: ImageDecoderProtocol | None = None
        self._preprocessor: object | None = None

    def _ensure_initialized(self) -> None:
        if self._decoder is None:
            from triton_shared_models.classification.preprocessing import CenterCropPreprocessor  # noqa: PLC0415

            self._decoder = create_decoder(DecoderType.OPENCV)
            self._preprocessor = CenterCropPreprocessor(crop_size=640)

    def __call__(self, event: ClassModel3Event) -> Model3Request | PreprocessingError | None:  # noqa: D102
        self._ensure_initialized()
        raw = _read_image_bytes(event)
        if raw is None or isinstance(raw, PreprocessingError):
            return raw
        decoded = _decode_image(self._decoder, raw, event.flow_meta.message_id)  # type: ignore[arg-type]
        if isinstance(decoded, PreprocessingError):
            return decoded

        tensor = self._preprocessor.preprocess(decoded)  # type: ignore[union-attr]
        return Model3Request(
            flow_meta=event.flow_meta,
            model_name=self._model_name,
            raw_image=tensor,
        )


class Model4Preprocessor(
    BaseProcessPoolProcessor[DetModel4Event, Model4Request | PreprocessingError],
):
    """1024x1024 TurboJPEG LetterBox preprocessor with shared memory."""

    def __init__(self, model_name: str, *, num_workers: int = 8) -> None:  # noqa: D107
        super().__init__(num_workers=num_workers, use_shm=True)
        self._model_name = model_name
        self._decoder: ImageDecoderProtocol | None = None
        self._preprocessor: object | None = None

    def _ensure_initialized(self) -> None:
        if self._decoder is None:
            from triton_shared_models.preprocessing import LetterBoxPreprocessor  # noqa: PLC0415

            self._decoder = create_decoder(DecoderType.TURBOJPEG)
            self._preprocessor = LetterBoxPreprocessor(target_size=1024)

    def __call__(self, event: DetModel4Event) -> Model4Request | PreprocessingError | None:  # noqa: D102
        self._ensure_initialized()
        return _preprocess_detection(event, self._model_name, self._decoder, self._preprocessor, Model4Request)  # type: ignore[arg-type,return-value]


def _do_inference(
    grpc_client: grpcclient.InferenceServerClient,
    event: TritonRequest,
) -> InferenceOutput | InferenceError | None:
    """Shared inference logic for all model branches."""
    start = time.monotonic()

    if event.raw_image is not None:
        dtype_str = "FP16" if event.raw_image.dtype == np.float16 else "FP32"
        input_tensor = grpcclient.InferInput(event.input_name, event.raw_image.shape, dtype_str)
        input_tensor.set_data_from_numpy(event.raw_image)
    elif event.image_bytes is not None:
        data = np.frombuffer(event.image_bytes, dtype=np.uint8)
        input_tensor = grpcclient.InferInput(event.input_name, [len(data)], "UINT8")
        input_tensor.set_data_from_numpy(data)
    else:
        return None

    outputs = [grpcclient.InferRequestedOutput(name) for name in event.output_names]

    try:
        result = grpc_client.infer(
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
    return InferenceOutput(
        flow_meta=event.flow_meta,
        model_name=event.model_name,
        model_version=event.model_version,
        outputs=output_arrays,
        inference_time_ms=TimeMetric.from_seconds(elapsed),
    )


def _wrap_det_inference(
    result: InferenceOutput | InferenceError | None,
    event: Model1Request | Model2Request | Model4Request,
    output_cls: type,
) -> object:
    """Wrap InferenceOutput into a per-model typed output with scale/pad."""
    if isinstance(result, InferenceOutput) and not isinstance(result, output_cls):
        return output_cls(
            flow_meta=result.flow_meta,
            model_name=result.model_name,
            model_version=result.model_version,
            outputs=result.outputs,
            inference_time_ms=result.inference_time_ms,
            preprocess_scale=event.preprocess_scale,
            preprocess_pad=event.preprocess_pad,
        )
    return result


class Model1InferenceNode(
    BaseThreadProcessor[Model1Request, Model1InfOut | InferenceError],
):
    """Inference wrapper for model 1 (640x640 detection)."""

    def __init__(self, grpc_url: str) -> None:  # noqa: D107
        super().__init__()
        self._client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=grpc_url)

    def __call__(self, event: Model1Request) -> Model1InfOut | InferenceError | None:  # noqa: D102
        result = _do_inference(self._client, event)
        return _wrap_det_inference(result, event, Model1InfOut)  # type: ignore[return-value]


class Model2InferenceNode(
    BaseThreadProcessor[Model2Request, Model2InfOut | InferenceError],
):
    """Inference wrapper for model 2 (608x960 FP16 detection)."""

    def __init__(self, grpc_url: str) -> None:  # noqa: D107
        super().__init__()
        self._client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=grpc_url)

    def __call__(self, event: Model2Request) -> Model2InfOut | InferenceError | None:  # noqa: D102
        result = _do_inference(self._client, event)
        return _wrap_det_inference(result, event, Model2InfOut)  # type: ignore[return-value]


class Model3InferenceNode(
    BaseThreadProcessor[Model3Request, Model3InfOut | InferenceError],
):
    """Inference wrapper for model 3 (640x640 classification)."""

    def __init__(self, grpc_url: str) -> None:  # noqa: D107
        super().__init__()
        self._client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=grpc_url)

    def __call__(self, event: Model3Request) -> Model3InfOut | InferenceError | None:  # noqa: D102
        result = _do_inference(self._client, event)
        if isinstance(result, InferenceOutput) and not isinstance(result, Model3InfOut):
            return Model3InfOut(
                flow_meta=result.flow_meta,
                model_name=result.model_name,
                model_version=result.model_version,
                outputs=result.outputs,
                inference_time_ms=result.inference_time_ms,
            )
        return result


class Model4InferenceNode(
    BaseThreadProcessor[Model4Request, Model4InfOut | InferenceError],
):
    """Inference wrapper for model 4 (1024x1024 detection with shmem)."""

    def __init__(self, grpc_url: str) -> None:  # noqa: D107
        super().__init__()
        self._client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=grpc_url)

    def __call__(self, event: Model4Request) -> Model4InfOut | InferenceError | None:  # noqa: D102
        result = _do_inference(self._client, event)
        return _wrap_det_inference(result, event, Model4InfOut)  # type: ignore[return-value]


_yolo_postprocessor = YoloPostprocessor()
_classification_postprocessor = ClassificationPostprocessor(top_k=5)


def _det_postprocess(event: Model1InfOut | Model2InfOut | Model4InfOut) -> DetectionResult:
    """Postprocess detection model output into DetectionResult."""
    detections = _yolo_postprocessor.postprocess(
        event.outputs,
        event.preprocess_scale,
        event.preprocess_pad,
    )
    return DetectionResult(
        flow_meta=event.flow_meta,
        detections=tuple(detections),
        model_name=event.model_name,
        inference_time_ms=event.inference_time_ms,
    )


def _cls_postprocess(event: Model3InfOut) -> ClassificationResult:
    """Postprocess classification model output into ClassificationResult."""
    predictions = _classification_postprocessor.postprocess(event.outputs)
    return ClassificationResult(
        flow_meta=event.flow_meta,
        predictions=tuple(predictions),
        model_name=event.model_name,
        inference_time_ms=event.inference_time_ms,
    )


class Model1PostProc(BaseThreadProcessor[Model1InfOut, DetectionResult]):
    """Postprocessor for model 1 detection results."""

    def __call__(self, event: Model1InfOut) -> DetectionResult | None:  # noqa: D102
        return _det_postprocess(event)


class Model2PostProc(BaseThreadProcessor[Model2InfOut, DetectionResult]):
    """Postprocessor for model 2 detection results."""

    def __call__(self, event: Model2InfOut) -> DetectionResult | None:  # noqa: D102
        return _det_postprocess(event)


class Model3PostProc(BaseThreadProcessor[Model3InfOut, ClassificationResult]):
    """Postprocessor for model 3 classification results."""

    def __call__(self, event: Model3InfOut) -> ClassificationResult | None:  # noqa: D102
        return _cls_postprocess(event)


class Model4PostProc(BaseThreadProcessor[Model4InfOut, DetectionResult]):
    """Postprocessor for model 4 detection results."""

    def __call__(self, event: Model4InfOut) -> DetectionResult | None:  # noqa: D102
        return _det_postprocess(event)
