"""Domain event models for the camera inference example pipeline.

CameraFrame represents a raw image captured from a camera.
DetectionResult represents inference results after processing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CameraFrame(BaseModel):
    """Raw image frame captured from a camera source."""

    model_config = ConfigDict(frozen=True, strict=True)

    camera_id: str
    frame_id: int = Field(ge=0)
    image_key: str
    timestamp_ms: int = Field(ge=0)


class DetectionResult(BaseModel):
    """Detection result after Triton inference processing."""

    model_config = ConfigDict(frozen=True, strict=True)

    camera_id: str
    frame_id: int = Field(ge=0)
    model_name: str
    detections_json: str
    inference_time_ms: float = Field(ge=0.0)
