"""Pipeline processors for the CV inference pipeline.

Router dispatches camera events to model-specific branches.
Postprocessor factories create typed callables for per-model results.
"""

from collections.abc import Callable

from loguru import logger
from triton_model_tools.inference.classification_postprocessing import ClassificationPostprocessor
from triton_model_tools.inference.postprocessing import YoloPostprocessor
from triton_redis_connector.engine.events import ClassificationResult, DetectionResult
from triton_redis_connector.inference.request import InferenceOutput
from typed_flow.protocols import BaseThreadProcessor

from reference_implementation.models import (
    CameraEvent,
    ClassModel3Event,
    DetModel1Event,
    DetModel2Event,
    DetModel4Event,
    RouteEvent,
)

CAM_MODEL1_AND_3 = {1, 2}
CAM_MODEL2 = {3, 4}
CAM_MODEL4 = {5, 6}

MODEL_MAP: dict[type[RouteEvent], str] = {
    DetModel1Event: "yolo11s",
    DetModel2Event: "yolo11s_608x960",
    ClassModel3Event: "cls_resnet18",
    DetModel4Event: "yolo11s",
}


class CameraRouterProcessor(
    BaseThreadProcessor[CameraEvent, RouteEvent],
):
    """Routes camera events to model-specific event types by cam_id.

    Injects ``model_name`` from ``MODEL_MAP`` so downstream preprocessors
    can look up metadata from Triton.
    """

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
                    model_name=MODEL_MAP[DetModel1Event],
                ),
                ClassModel3Event(
                    flow_meta=event.flow_meta,
                    image_path=event.image_path,
                    image_bytes=event.image_bytes,
                    cam_id=event.cam_id,
                    model_name=MODEL_MAP[ClassModel3Event],
                ),
            ]
        if event.cam_id in CAM_MODEL2:
            return DetModel2Event(
                flow_meta=event.flow_meta,
                image_path=event.image_path,
                image_bytes=event.image_bytes,
                cam_id=event.cam_id,
                model_name=MODEL_MAP[DetModel2Event],
            )
        if event.cam_id in CAM_MODEL4:
            return DetModel4Event(
                flow_meta=event.flow_meta,
                image_path=event.image_path,
                image_bytes=event.image_bytes,
                cam_id=event.cam_id,
                model_name=MODEL_MAP[DetModel4Event],
            )
        logger.warning("Unknown cam_id={}, dropping event", event.cam_id)
        return None


_yolo_postprocessor = YoloPostprocessor()
_classification_postprocessor = ClassificationPostprocessor(top_k=5)


def make_det_postprocessor[TIn: InferenceOutput, TResult: DetectionResult](
    _input_cls: type[TIn],
    result_cls: type[TResult],
) -> Callable[[TIn], TResult]:
    """Create a typed detection postprocessor callable for a specific result type."""

    def postprocess(event: TIn) -> TResult:
        detections = _yolo_postprocessor.postprocess(
            event.outputs,
            event.preprocess_scale,
            event.preprocess_pad,
        )
        return result_cls(
            flow_meta=event.flow_meta,
            detections=tuple(detections),
            model_name=event.model_name,
            inference_time_ms=event.inference_time_ms,
        )

    return postprocess


def make_cls_postprocessor[TIn: InferenceOutput, TResult: ClassificationResult](
    _input_cls: type[TIn],
    result_cls: type[TResult],
) -> Callable[[TIn], TResult]:
    """Create a typed classification postprocessor callable for a specific result type."""

    def postprocess(event: TIn) -> TResult:
        predictions = _classification_postprocessor.postprocess(event.outputs)
        return result_cls(
            flow_meta=event.flow_meta,
            predictions=tuple(predictions),
            model_name=event.model_name,
            inference_time_ms=event.inference_time_ms,
        )

    return postprocess
