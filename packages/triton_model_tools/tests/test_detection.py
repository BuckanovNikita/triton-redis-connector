"""Tests for detection result models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from triton_model_tools.models.detection import (
    Detection,
    FolderInferenceResult,
    ImageDetectionResult,
)


class TestDetection:
    def test_rejects_negative_class_id(self) -> None:
        with pytest.raises(ValidationError):
            Detection(x1=0, y1=0, x2=1, y2=1, confidence=0.5, class_id=-1)

    def test_accepts_confidence_above_one(self) -> None:
        det = Detection(x1=0, y1=0, x2=1, y2=1, confidence=1.5, class_id=0)
        assert det.confidence == 1.5

    def test_rejects_confidence_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            Detection(x1=0, y1=0, x2=1, y2=1, confidence=-0.1, class_id=0)


class TestImageDetectionResult:
    def test_rejects_negative_inference_time(self) -> None:
        with pytest.raises(ValidationError):
            ImageDetectionResult(
                image_path="images/img.jpg",
                model_name="yolo11s",
                detections=[],
                inference_time_ms=-1.0,
            )


class TestFolderInferenceResult:
    def test_rejects_negative_total_time(self) -> None:
        with pytest.raises(ValidationError):
            FolderInferenceResult(
                model_name="yolo11s",
                results=[],
                total_time_s=-1.0,
                avg_time_ms=0.0,
            )

    def test_rejects_negative_avg_time(self) -> None:
        with pytest.raises(ValidationError):
            FolderInferenceResult(
                model_name="yolo11s",
                results=[],
                total_time_s=0.0,
                avg_time_ms=-1.0,
            )
