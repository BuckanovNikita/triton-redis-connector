"""Tests for pipeline predict functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from triton_model_tools.detection import Detection, FolderInferenceResult, ImageDetectionResult
from triton_model_tools.pipeline_predict import run_pipeline_predict

if TYPE_CHECKING:
    from pathlib import Path


def _make_folder_result() -> FolderInferenceResult:
    return FolderInferenceResult(
        model_name="yolo11s",
        results=[
            ImageDetectionResult(
                image_path="image_0001",
                model_name="yolo11s",
                detections=[
                    Detection(class_id=0, x1=10.0, y1=20.0, x2=100.0, y2=200.0, confidence=0.95),
                ],
                inference_time_ms=15.0,
            ),
            ImageDetectionResult(
                image_path="image_0002",
                model_name="yolo11s",
                detections=[],
                inference_time_ms=12.0,
            ),
        ],
        total_time_s=0.5,
        avg_time_ms=250.0,
    )


class TestRunPipelinePredict:
    @patch("triton_model_tools.pipeline_predict.ConnectorE2ETestHarness")
    def test_saves_txt_files(self, mock_harness_cls: MagicMock, tmp_path: Path) -> None:
        mock_harness = MagicMock()
        mock_harness.run_through_connector.return_value = _make_folder_result()
        mock_harness_cls.return_value = mock_harness

        from triton_model_tools.e2e_test import E2ETestConfig

        config = E2ETestConfig(model_name="yolo11s")
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        output_dir = tmp_path / "results"

        result = run_pipeline_predict(image_dir, output_dir, config)

        assert len(result.results) == 2
        assert (output_dir / "image_0001.txt").exists()
        assert (output_dir / "image_0002.txt").exists()

        content = (output_dir / "image_0001.txt").read_text().strip()
        assert content.startswith("0 ")
        assert "image_0002.txt" in [f.name for f in output_dir.iterdir()]

    @patch("triton_model_tools.pipeline_predict.ConnectorE2ETestHarness")
    def test_creates_output_dir(self, mock_harness_cls: MagicMock, tmp_path: Path) -> None:
        mock_harness = MagicMock()
        mock_harness.run_through_connector.return_value = _make_folder_result()
        mock_harness_cls.return_value = mock_harness

        from triton_model_tools.e2e_test import E2ETestConfig

        config = E2ETestConfig(model_name="yolo11s")
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        output_dir = tmp_path / "nested" / "output"

        run_pipeline_predict(image_dir, output_dir, config)

        assert output_dir.exists()
