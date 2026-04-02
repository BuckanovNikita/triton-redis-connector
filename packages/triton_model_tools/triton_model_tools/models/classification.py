"""Pydantic models for classification results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ClassificationPrediction(BaseModel):
    """Single classification prediction with class ID, name, and probability."""

    model_config = ConfigDict(frozen=True)

    class_id: int = Field(ge=0)
    class_name: str = ""
    probability: float = Field(ge=0.0, le=1.0)


class ImageClassificationResult(BaseModel):
    """Classification results for a single image."""

    model_config = ConfigDict(frozen=True)

    image_path: str
    model_name: str
    predictions: list[ClassificationPrediction]
    inference_time_ms: float = Field(ge=0.0)


class FolderClassificationResult(BaseModel):
    """Aggregated classification results for a batch of images."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    results: list[ImageClassificationResult]
    total_time_s: float = Field(ge=0.0)
    avg_time_ms: float = Field(ge=0.0)
