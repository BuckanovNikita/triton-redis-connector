"""Event models for the CV inference pipeline.

Camera events are read from Redis, routed by cam_id to typed per-model
events, and processed through detection/classification branches.
"""

from typing import ClassVar

from triton_redis_connector.engine.events import (
    ClassificationResult,
    DetectionResult,
    ImagePayload,
)
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


RouteEvent = DetModel1Event | ClassModel3Event | DetModel2Event | DetModel4Event
ResultEvent = DetectionResult | ClassificationResult

__all__ = [
    "CameraEvent",
    "ClassModel3Event",
    "DetModel1Event",
    "DetModel2Event",
    "DetModel4Event",
    "ResultEvent",
    "RouteEvent",
]
