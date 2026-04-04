"""Pydantic models for YOLO detection results."""

from pydantic import BaseModel, ConfigDict, Field
from triton_shared_models.detection import Detection
from typed_flow.protocols.metrics import TimeMetric

__all__ = ["Detection", "FolderInferenceResult", "ImageDetectionResult"]


class ImageDetectionResult(BaseModel):
    """Detection results for a single image."""

    model_config = ConfigDict(frozen=True)

    image_path: str
    model_name: str
    detections: list[Detection]
    inference_time_ms: TimeMetric = TimeMetric(0)


class FolderInferenceResult(BaseModel):
    """Aggregated detection results for a batch of images."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    results: list[ImageDetectionResult]
    total_time_s: float = Field(ge=0.0)
    avg_time_ms: float = Field(ge=0.0)
