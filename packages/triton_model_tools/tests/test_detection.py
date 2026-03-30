"""Tests for detection result models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from triton_model_tools.detection import (
    Detection,
    FolderInferenceResult,
    ImageDetectionResult,
)


class TestDetection:
    def test_creates_detection(self) -> None:
        det = Detection(
            x1=10.0,
            y1=20.0,
            x2=100.0,
            y2=200.0,
            confidence=0.95,
            class_id=0,
            class_name="person",
        )
        assert det.x1 == 10.0
        assert det.y1 == 20.0
        assert det.x2 == 100.0
        assert det.y2 == 200.0
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == "person"

    def test_default_class_name_is_empty(self) -> None:
        det = Detection(x1=0, y1=0, x2=1, y2=1, confidence=0.5, class_id=1)
        assert det.class_name == ""

    def test_frozen_model(self) -> None:
        det = Detection(x1=0, y1=0, x2=1, y2=1, confidence=0.5, class_id=0)
        with pytest.raises(ValidationError):
            det.x1 = 5.0  # type: ignore[misc]

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
    def test_creates_result_with_detections(self) -> None:
        det = Detection(x1=0, y1=0, x2=50, y2=50, confidence=0.9, class_id=0)
        result = ImageDetectionResult(
            image_path="images/img.jpg",
            model_name="yolo11s",
            detections=[det],
            inference_time_ms=15.3,
        )
        assert result.image_path == "images/img.jpg"
        assert result.model_name == "yolo11s"
        assert len(result.detections) == 1
        assert result.inference_time_ms == 15.3

    def test_creates_result_with_empty_detections(self) -> None:
        result = ImageDetectionResult(
            image_path="images/empty.jpg",
            model_name="yolo11s",
            detections=[],
            inference_time_ms=5.0,
        )
        assert len(result.detections) == 0

    def test_frozen_model(self) -> None:
        result = ImageDetectionResult(
            image_path="images/img.jpg",
            model_name="yolo11s",
            detections=[],
            inference_time_ms=5.0,
        )
        with pytest.raises(ValidationError):
            result.model_name = "other"  # type: ignore[misc]

    def test_rejects_negative_inference_time(self) -> None:
        with pytest.raises(ValidationError):
            ImageDetectionResult(
                image_path="images/img.jpg",
                model_name="yolo11s",
                detections=[],
                inference_time_ms=-1.0,
            )


class TestFolderInferenceResult:
    def test_creates_folder_result(self) -> None:
        img_result = ImageDetectionResult(
            image_path="images/img1.jpg",
            model_name="yolo11s",
            detections=[],
            inference_time_ms=10.0,
        )
        folder_result = FolderInferenceResult(
            model_name="yolo11s",
            results=[img_result],
            total_time_s=0.5,
            avg_time_ms=10.0,
        )
        assert folder_result.model_name == "yolo11s"
        assert len(folder_result.results) == 1
        assert folder_result.total_time_s == 0.5
        assert folder_result.avg_time_ms == 10.0

    def test_frozen_model(self) -> None:
        folder_result = FolderInferenceResult(
            model_name="yolo11s",
            results=[],
            total_time_s=0.0,
            avg_time_ms=0.0,
        )
        with pytest.raises(ValidationError):
            folder_result.model_name = "other"  # type: ignore[misc]

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
