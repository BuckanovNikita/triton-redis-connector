"""Domain event models for the reference implementation pipelines.

Multi-task pipeline (detection + classification):
  ImageTaskInput
    -> TaskRouter
      -> [detect]   PipelineEvent[TaskMeta, DetectionImagePayload]
      -> [classify] PipelineEvent[TaskMeta, ClassificationImagePayload]

Dual-model steel inspection pipeline:
  SteelInspectionInput
    -> CameraBranchRouter
    -> PipelineEvent[SteelMeta, EvenImagePayload | OddImagePayload]
    -> ...processing stages...
    -> PipelineEvent[SteelMeta, EvenDetectionResult | OddDetectionResult]

All user models are plain BaseModel subclasses. Framework metadata
lives on PipelineEvent.fw, not on user models.
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict
from triton_redis_connector.engine.events import (
    DetectionResult,
    ImagePayload,
)
from typed_flow.protocols import Counter, GroupBy

DETECTION_MODEL = "yolo26l"
CLASSIFICATION_MODEL = "resnet18"

DETECTION_TARGET_SIZE = (640, 640)
CLASSIFICATION_TARGET_SIZE = (224, 224)

EVEN_MODEL = "yolo11s_960x608"
ODD_MODEL = "yolo26l"


class TaskType(StrEnum):
    """Supported inference task types."""

    DETECT = auto()
    CLASSIFY = auto()


class TaskMeta(BaseModel):
    """Typed context carrying task metadata through the pipeline.

    Travels inside ``PipelineEvent.context`` through all processing stages.
    The ``task_type`` field determines which branch processed the image.
    """

    model_config = ConfigDict(frozen=True)

    _metrics_: ClassVar[list[Counter]] = [Counter("task_received")]

    task_type: TaskType
    image_path: str
    source_id: Annotated[str, GroupBy("source_events", window_s=60)]
    priority: int = 0


class ImageTaskInput(BaseModel):
    """Raw task event parsed from a Redis stream envelope.

    The ``task_type`` field determines whether the image is routed to
    detection or classification.
    """

    model_config = ConfigDict(frozen=True, strict=True)

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


class SteelMeta(BaseModel):
    """Typed context carrying domain data through the pipeline.

    Travels inside ``PipelineEvent.context`` through all processing stages.
    """

    model_config = ConfigDict(frozen=True)

    _metrics_: ClassVar[list[Counter]] = [Counter("inspection_received")]

    receipt_ts: float
    camera_no: Annotated[int, GroupBy("camera_no_per_minute", window_s=60)]
    image_path: str
    frame_counter: int
    metal_type: str


class SteelInspectionInput(BaseModel):
    """Raw inspection event parsed from a Redis stream envelope."""

    model_config = ConfigDict(frozen=True, strict=True)

    receipt_ts: float
    camera_no: int
    image_path: str
    frame_counter: int
    metal_type: str | None = None


class EvenImagePayload(ImagePayload):
    """Image payload for even camera_no frames (yolo11s_960x608)."""

    _metrics_: ClassVar[list[Counter]] = [Counter("even_camera_frame")]


class OddImagePayload(ImagePayload):
    """Image payload for odd camera_no frames (yolo26l)."""

    _metrics_: ClassVar[list[Counter]] = [Counter("odd_camera_frame")]


class EvenDetectionResult(DetectionResult):
    """Detection result from the even-camera model (yolo11s_960x608)."""

    _metrics_: ClassVar[list[Counter]] = [Counter("even_model_result")]


class OddDetectionResult(DetectionResult):
    """Detection result from the odd-camera model (yolo26l)."""

    _metrics_: ClassVar[list[Counter]] = [Counter("odd_model_result")]
