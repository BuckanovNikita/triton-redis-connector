"""Custom pipeline processors for the reference implementation.

Contains routers for both the multi-task and dual-model pipelines.
Custom sinks are replaced by generic TypedRedisStreamSink.
"""

from __future__ import annotations

from loguru import logger
from triton_redis_connector.engine.events import DetectionResult
from typed_flow.engine.events import PipelineEvent, Unprocessed
from typed_flow.protocols import BaseThreadProcessor

from reference_implementation.events import (
    EVEN_MODEL,
    ClassificationImagePayload,
    DetectionImagePayload,
    EvenDetectionResult,
    EvenImagePayload,
    ImageTaskInput,
    OddDetectionResult,
    OddImagePayload,
    SteelInspectionInput,
    SteelMeta,
    TaskMeta,
    TaskType,
)


class TaskRouter(
    BaseThreadProcessor[
        PipelineEvent[ImageTaskInput, Unprocessed],
        list[PipelineEvent[TaskMeta, DetectionImagePayload] | PipelineEvent[TaskMeta, ClassificationImagePayload]],
    ],
):
    """Routes incoming tasks to detection or classification branches.

    Parses the ``task_type`` field and produces a PipelineEvent with
    the appropriate typed payload for downstream preprocessors.
    """

    def __init__(self) -> None:
        """Initialize the task router."""
        super().__init__()

    def __call__(
        self, event: PipelineEvent[ImageTaskInput, Unprocessed]
    ) -> list[PipelineEvent[TaskMeta, DetectionImagePayload] | PipelineEvent[TaskMeta, ClassificationImagePayload]]:
        """Route event to detection or classification branch based on task type."""
        inp = event.context
        try:
            task_type = TaskType(inp.task_type)
        except ValueError:
            logger.error("Unknown task_type={}, dropping event", inp.task_type)
            return []

        context = TaskMeta(
            task_type=task_type,
            image_path=inp.image_path,
            source_id=inp.source_id,
            priority=inp.priority,
        )

        return [
            PipelineEvent[TaskMeta, DetectionImagePayload](
                fw=event.fw,
                context=context,
                payload=DetectionImagePayload(image_path=inp.image_path),
            ),
            PipelineEvent[TaskMeta, ClassificationImagePayload](
                fw=event.fw,
                context=context,
                payload=ClassificationImagePayload(image_path=inp.image_path),
            ),
        ]


class CameraBranchRouter(
    BaseThreadProcessor[
        PipelineEvent[SteelInspectionInput, Unprocessed],
        PipelineEvent[SteelMeta, EvenImagePayload] | PipelineEvent[SteelMeta, OddImagePayload],
    ],
):
    """Routes inspection input to even/odd camera preprocessing paths.

    Even camera_no -> PipelineEvent[SteelMeta, EvenImagePayload]
    Odd camera_no  -> PipelineEvent[SteelMeta, OddImagePayload]
    """

    def __init__(self) -> None:
        """Initialize the camera branch router."""
        super().__init__()

    def __call__(
        self, event: PipelineEvent[SteelInspectionInput, Unprocessed]
    ) -> PipelineEvent[SteelMeta, EvenImagePayload] | PipelineEvent[SteelMeta, OddImagePayload] | None:
        """Route event to even or odd camera branch based on camera number."""
        inp = event.context
        metal_type = inp.metal_type or ("HK" if inp.camera_no % 2 == 0 else "GK")
        context = SteelMeta(
            receipt_ts=inp.receipt_ts,
            camera_no=inp.camera_no,
            image_path=inp.image_path,
            frame_counter=inp.frame_counter,
            metal_type=metal_type,
        )

        if inp.camera_no % 2 == 0:
            return PipelineEvent[SteelMeta, EvenImagePayload](
                fw=event.fw,
                context=context,
                payload=EvenImagePayload(image_path=inp.image_path),
            )

        return PipelineEvent(
            fw=event.fw,
            context=context,
            payload=OddImagePayload(image_path=inp.image_path),
        )


class ModelResultRouter(
    BaseThreadProcessor[
        PipelineEvent[SteelMeta, DetectionResult],
        PipelineEvent[SteelMeta, EvenDetectionResult] | PipelineEvent[SteelMeta, OddDetectionResult],
    ],
):
    """Routes detection results to per-model typed sinks based on model_name.

    Converts generic DetectionResult payload into model-specific subclasses
    so the type-based routing table sends results to the correct sink.
    """

    def __init__(self) -> None:
        """Initialize the model result router."""
        super().__init__()

    def __call__(
        self, event: PipelineEvent[SteelMeta, DetectionResult]
    ) -> PipelineEvent[SteelMeta, EvenDetectionResult] | PipelineEvent[SteelMeta, OddDetectionResult] | None:
        """Route detection result to even or odd typed sink by model name."""
        result = event.payload
        if result.model_name == EVEN_MODEL:
            typed_result = EvenDetectionResult(
                detections=result.detections,
                model_name=result.model_name,
                inference_time_ms=result.inference_time_ms,
            )
            return PipelineEvent[SteelMeta, EvenDetectionResult](
                fw=event.fw, context=event.context, payload=typed_result
            )

        typed_result_odd = OddDetectionResult(
            detections=result.detections,
            model_name=result.model_name,
            inference_time_ms=result.inference_time_ms,
        )
        return PipelineEvent[SteelMeta, OddDetectionResult](
            fw=event.fw, context=event.context, payload=typed_result_odd
        )
