"""Tests for FiftyOne export integration (mocked)."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from triton_model_tools.benchmarks.fiftyone_export import (
    FIFTYONE_INSTALL_MSG,
    export_to_fiftyone,
    launch_fiftyone_app,
)
from triton_model_tools.models.detection import Detection, ImageDetectionResult

MOCK_DIMENSIONS = (640, 480)
_DIMS_PATCH = "triton_model_tools.benchmarks.fiftyone_export._read_image_dimensions"


def _sample_results() -> list[ImageDetectionResult]:
    return [
        ImageDetectionResult(
            image_path="/images/test.jpg",
            model_name="yolo11s",
            detections=[
                Detection(x1=10, y1=20, x2=100, y2=200, confidence=0.9, class_id=0, class_name="person"),
                Detection(x1=50, y1=60, x2=150, y2=250, confidence=0.8, class_id=1, class_name="car"),
            ],
            inference_time_ms=15.0,
        ),
    ]


class TestExportToFiftyone:
    def test_raises_import_error_when_fiftyone_missing(self) -> None:
        with (
            patch.dict(sys.modules, {"fiftyone": None}),
            pytest.raises(ImportError, match=FIFTYONE_INSTALL_MSG),
        ):
            export_to_fiftyone(_sample_results(), "test-dataset")

    def test_creates_dataset_with_mock(self) -> None:
        mock_fo = MagicMock()
        mock_dataset = MagicMock()
        mock_fo.Dataset.return_value = mock_dataset

        with (
            patch.dict(sys.modules, {"fiftyone": mock_fo}),
            patch(_DIMS_PATCH, return_value=MOCK_DIMENSIONS),
        ):
            export_to_fiftyone(_sample_results(), "test-dataset")

        mock_fo.Dataset.assert_called_once_with(name="test-dataset", overwrite=True)
        mock_dataset.add_samples.assert_called_once()

    def test_creates_correct_number_of_samples(self) -> None:
        mock_fo = MagicMock()
        mock_dataset = MagicMock()
        mock_fo.Dataset.return_value = mock_dataset

        results = [
            ImageDetectionResult(
                image_path=f"/images/img{i}.jpg",
                model_name="yolo11s",
                detections=[],
                inference_time_ms=5.0,
            )
            for i in range(3)
        ]

        with (
            patch.dict(sys.modules, {"fiftyone": mock_fo}),
            patch(_DIMS_PATCH, return_value=MOCK_DIMENSIONS),
        ):
            export_to_fiftyone(results, "multi-image")

        samples_arg = mock_dataset.add_samples.call_args[0][0]
        assert len(samples_arg) == 3


class TestLaunchFiftyoneApp:
    def test_raises_import_error_when_fiftyone_missing(self) -> None:
        with (
            patch.dict(sys.modules, {"fiftyone": None}),
            pytest.raises(ImportError, match=FIFTYONE_INSTALL_MSG),
        ):
            launch_fiftyone_app(_sample_results(), "test-dataset")

    def test_launches_app_with_mock(self) -> None:
        mock_fo = MagicMock()
        mock_dataset = MagicMock()
        mock_fo.Dataset.return_value = mock_dataset
        mock_fo.load_dataset.return_value = mock_dataset
        mock_session = MagicMock()
        mock_fo.launch_app.return_value = mock_session

        with (
            patch.dict(sys.modules, {"fiftyone": mock_fo}),
            patch(_DIMS_PATCH, return_value=MOCK_DIMENSIONS),
        ):
            launch_fiftyone_app(_sample_results(), "launch-test")

        mock_fo.load_dataset.assert_called_once_with("launch-test")
        mock_fo.launch_app.assert_called_once_with(mock_dataset)
        mock_session.wait.assert_called_once()
