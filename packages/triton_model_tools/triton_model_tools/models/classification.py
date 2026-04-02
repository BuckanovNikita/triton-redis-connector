"""Pydantic models for classification results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from triton_shared_models.classification import ClassificationPrediction

__all__ = [
    "ClassificationPrediction",
    "FolderClassificationResult",
    "ImageClassificationResult",
]


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
