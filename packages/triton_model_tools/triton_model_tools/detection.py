"""Pydantic models for YOLO detection results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    """Single object detection with bounding box, confidence, and class info."""

    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int = Field(ge=0)
    class_name: str = ""


class ImageDetectionResult(BaseModel):
    """Detection results for a single image."""

    model_config = ConfigDict(frozen=True)

    image_path: str
    model_name: str
    detections: list[Detection]
    inference_time_ms: float = Field(ge=0.0)


class FolderInferenceResult(BaseModel):
    """Aggregated detection results for a batch of images."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    results: list[ImageDetectionResult]
    total_time_s: float = Field(ge=0.0)
    avg_time_ms: float = Field(ge=0.0)
