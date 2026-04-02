"""Domain event models for the multi-task pipeline (detection + classification).

Events flow through two branches based on task_type:
  ImageTaskInput
    -> TaskRouter
      -> [detect]   PipelineEvent[TaskMeta, DetectionImagePayload]
                       -> ClientPreprocessorNode -> InferenceNode -> PostProcessorNode
                       -> PipelineEvent[TaskMeta, DetectionResult]
                       -> DetectionResultSink -> task:detection_results
      -> [classify] PipelineEvent[TaskMeta, ClassificationImagePayload]
                       -> ClientPreprocessorNode -> InferenceNode -> PostProcessorNode
                       -> PipelineEvent[TaskMeta, ClassificationResult]
                       -> ClassificationResultSink -> task:classification_results

Per-task ImagePayload subclasses split the preprocessing path.
DetectionResult and ClassificationResult are already distinct types
from engine.events, so sink routing works without additional subclasses.
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Annotated, ClassVar

from triton_redis_connector.engine.events import Event, ImagePayload
from triton_redis_connector.protocols import Counter, GroupBy

DETECTION_MODEL = "yolo26l"
CLASSIFICATION_MODEL = "resnet18"

DETECTION_TARGET_SIZE = (640, 640)
CLASSIFICATION_TARGET_SIZE = (224, 224)


class TaskType(StrEnum):
    """Supported inference task types."""

    DETECT = auto()
    CLASSIFY = auto()


class TaskMeta(Event):
    """Typed context carrying task metadata through the pipeline.

    Travels inside ``PipelineEvent.context`` through all processing stages.
    The ``task_type`` field determines which branch processed the image.
    """

    _metrics_: ClassVar[list[Counter]] = [Counter("task_received")]

    task_type: TaskType
    image_path: str
    source_id: Annotated[str, GroupBy("source_events", window_s=60)]
    priority: int = 0


class ImageTaskInput(Event):
    """Raw task event parsed from a Redis stream envelope.

    The ``task_type`` field determines whether the image is routed to
    detection or classification.
    """

    task_type: str
    image_path: str
    source_id: str = ""
    priority: int = 0


class DetectionImagePayload(ImagePayload):
    """Image payload for the detection branch (YOLO)."""

    _metrics_: ClassVar[list[Counter]] = [Counter("detection_image")]


class ClassificationImagePayload(ImagePayload):
    """Image payload for the classification branch (ResNet18)."""

    _metrics_: ClassVar[list[Counter]] = [Counter("classification_image")]
