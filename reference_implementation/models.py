"""Event models for the CV inference pipeline.

Camera events are read from Redis, routed by cam_id to typed per-model
events, and processed through detection/classification branches.
Per-model marker types for TritonRequest, InferenceOutput, and Result
enable fully type-driven routing through the pipeline.
"""

from typing import ClassVar

from triton_redis_connector.engine.events import (
    ClassificationResult,
    DetectionResult,
    ImagePayload,
    TritonRequest,
)
from triton_redis_connector.inference.request import InferenceOutput
from typed_flow.protocols import Counter
from typed_flow.protocols.messaging import BaseEvent


class CameraEvent(BaseEvent):
    """Raw camera event from Redis stream."""

    _metrics_: ClassVar[list[Counter]] = [Counter("camera_events_received")]

    cam_id: int
    image_path: str = ""
    image_bytes: bytes | None = None


class DetModel1Event(ImagePayload):
    """640x640 TurboJPEG detection event (cams 1-2)."""

    cam_id: int


class ClassModel3Event(ImagePayload):
    """640x640 classification event (cams 1-2)."""

    cam_id: int


class DetModel2Event(ImagePayload):
    """608x960 FP16 OpenCV detection event (cams 3-4)."""

    cam_id: int


class DetModel4Event(ImagePayload):
    """1024x1024 TurboJPEG stream-bytes detection event (cams 5-6)."""

    cam_id: int


# Per-model request markers
class DetModel1Request(TritonRequest):
    """Triton request marker for detection model 1."""


class DetModel2Request(TritonRequest):
    """Triton request marker for detection model 2."""


class ClassModel3Request(TritonRequest):
    """Triton request marker for classification model 3."""


class DetModel4Request(TritonRequest):
    """Triton request marker for detection model 4."""


# Per-model inference output markers
class DetModel1Output(InferenceOutput):
    """Inference output marker for detection model 1."""


class DetModel2Output(InferenceOutput):
    """Inference output marker for detection model 2."""


class ClassModel3Output(InferenceOutput):
    """Inference output marker for classification model 3."""


class DetModel4Output(InferenceOutput):
    """Inference output marker for detection model 4."""


# Per-model result markers
class DetModel1Result(DetectionResult):
    """Result marker for detection model 1."""


class DetModel2Result(DetectionResult):
    """Result marker for detection model 2."""


class ClassModel3Result(ClassificationResult):
    """Result marker for classification model 3."""


class DetModel4Result(DetectionResult):
    """Result marker for detection model 4."""


RouteEvent = DetModel1Event | ClassModel3Event | DetModel2Event | DetModel4Event
ResultEvent = DetModel1Result | DetModel2Result | ClassModel3Result | DetModel4Result

__all__ = [
    "CameraEvent",
    "ClassModel3Event",
    "ClassModel3Output",
    "ClassModel3Request",
    "ClassModel3Result",
    "DetModel1Event",
    "DetModel1Output",
    "DetModel1Request",
    "DetModel1Result",
    "DetModel2Event",
    "DetModel2Output",
    "DetModel2Request",
    "DetModel2Result",
    "DetModel4Event",
    "DetModel4Output",
    "DetModel4Request",
    "DetModel4Result",
    "ResultEvent",
    "RouteEvent",
]
