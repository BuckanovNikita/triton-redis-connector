"""Pipeline processors for the CV inference pipeline.

Router dispatches camera events to model-specific branches.
Postprocessor callables transform inference outputs into results.
"""

from loguru import logger
from triton_model_tools.inference.classification_postprocessing import ClassificationPostprocessor
from triton_model_tools.inference.postprocessing import YoloPostprocessor
from triton_redis_connector.engine.events import ClassificationResult, DetectionResult
from triton_redis_connector.inference.request import ClassificationInferenceOutput, DetectionInferenceOutput
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


_yolo_postprocessor = YoloPostprocessor()
_classification_postprocessor = ClassificationPostprocessor(top_k=5)


def det_postprocess(event: DetectionInferenceOutput) -> DetectionResult:
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


def cls_postprocess(event: ClassificationInferenceOutput) -> ClassificationResult:
    """Postprocess classification model output into ClassificationResult."""
    predictions = _classification_postprocessor.postprocess(event.outputs)
    return ClassificationResult(
        flow_meta=event.flow_meta,
        predictions=tuple(predictions),
        model_name=event.model_name,
        inference_time_ms=event.inference_time_ms,
    )
