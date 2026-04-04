"""Event models for the CV inference pipeline.

Camera events are read from Redis, routed by cam_id to typed per-model
events, processed through model-specific preprocessors and inference,
and written as detection/classification results.
"""

from typing import ClassVar

from pydantic import ConfigDict
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


class Model1Request(TritonRequest):
    """TritonRequest carrying letterbox scale/pad for model 1 postprocessing."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    preprocess_scale: float = 1.0
    preprocess_pad: tuple[int, int] = (0, 0)


class Model2Request(TritonRequest):
    """TritonRequest carrying letterbox scale/pad for model 2 postprocessing."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    preprocess_scale: float = 1.0
    preprocess_pad: tuple[int, int] = (0, 0)


class Model3Request(TritonRequest):
    """TritonRequest for classification model 3 (no scale/pad needed)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class Model4Request(TritonRequest):
    """TritonRequest carrying letterbox scale/pad for model 4 postprocessing."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    preprocess_scale: float = 1.0
    preprocess_pad: tuple[int, int] = (0, 0)


class Model1InfOut(InferenceOutput):
    """InferenceOutput for model 1 with preprocessing metadata."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    preprocess_scale: float = 1.0
    preprocess_pad: tuple[int, int] = (0, 0)


class Model2InfOut(InferenceOutput):
    """InferenceOutput for model 2 with preprocessing metadata."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    preprocess_scale: float = 1.0
    preprocess_pad: tuple[int, int] = (0, 0)


class Model3InfOut(InferenceOutput):
    """InferenceOutput for classification model 3."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class Model4InfOut(InferenceOutput):
    """InferenceOutput for model 4 with preprocessing metadata."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    preprocess_scale: float = 1.0
    preprocess_pad: tuple[int, int] = (0, 0)


Model1Request.model_rebuild()
Model2Request.model_rebuild()
Model3Request.model_rebuild()
Model4Request.model_rebuild()
Model1InfOut.model_rebuild()
Model2InfOut.model_rebuild()
Model3InfOut.model_rebuild()
Model4InfOut.model_rebuild()

RouteEvent = DetModel1Event | ClassModel3Event | DetModel2Event | DetModel4Event
ResultEvent = DetectionResult | ClassificationResult

__all__ = [
    "CameraEvent",
    "ClassModel3Event",
    "DetModel1Event",
    "DetModel2Event",
    "DetModel4Event",
    "Model1InfOut",
    "Model1Request",
    "Model2InfOut",
    "Model2Request",
    "Model3InfOut",
    "Model3Request",
    "Model4InfOut",
    "Model4Request",
    "ResultEvent",
    "RouteEvent",
]
