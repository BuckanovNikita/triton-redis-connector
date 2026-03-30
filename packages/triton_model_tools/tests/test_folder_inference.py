"""Tests for folder-level batch inference."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from triton_model_tools.detection import FolderInferenceResult
from triton_model_tools.folder_inference import (
    IMAGE_EXTENSIONS,
    _collect_image_paths,
    run_folder_inference,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestCollectImagePaths:
    def test_finds_jpg_and_png(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "b.png").write_bytes(b"fake")
        (tmp_path / "c.txt").write_text("not an image")

        paths = _collect_image_paths(tmp_path)

        assert len(paths) == 2
        suffixes = {p.suffix for p in paths}
        assert suffixes == {".jpg", ".png"}

    def test_sorted_alphabetically(self, tmp_path: Path) -> None:
        (tmp_path / "z.jpg").write_bytes(b"fake")
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "m.jpg").write_bytes(b"fake")

        paths = _collect_image_paths(tmp_path)

        assert [p.name for p in paths] == ["a.jpg", "m.jpg", "z.jpg"]

    def test_raises_on_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No image files found"):
            _collect_image_paths(tmp_path)

    def test_supported_extensions(self) -> None:
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".bmp" in IMAGE_EXTENSIONS
        assert ".tiff" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS


class TestRunFolderInference:
    @patch("triton_model_tools.folder_inference._load_image")
    @patch("triton_model_tools.folder_inference.TritonInferenceClient")
    def test_processes_all_images(
        self,
        mock_client_cls: MagicMock,
        mock_load_image: MagicMock,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "img1.jpg").write_bytes(b"fake")
        (tmp_path / "img2.jpg").write_bytes(b"fake")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_load_image.return_value = fake_image

        mock_client = MagicMock()
        mock_client.infer_raw.return_value = {
            "output0": np.zeros((1, 84, 8400), dtype=np.float32),
        }
        mock_client_cls.return_value = mock_client

        result = run_folder_inference(
            image_dir=tmp_path,
            model_name="yolo11s",
            triton_url="localhost:8001",
        )

        assert isinstance(result, FolderInferenceResult)
        assert len(result.results) == 2
        assert result.model_name == "yolo11s"
        assert result.total_time_s > 0
        assert result.avg_time_ms > 0
        assert mock_client.infer_raw.call_count == 2

    @patch("triton_model_tools.folder_inference._load_image")
    @patch("triton_model_tools.folder_inference.TritonInferenceClient")
    def test_end2end_mode(
        self,
        mock_client_cls: MagicMock,
        mock_load_image: MagicMock,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "img.jpg").write_bytes(b"fake")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_load_image.return_value = fake_image

        mock_client = MagicMock()
        end2end_output = np.array(
            [[[100, 100, 200, 200, 0.9, 0]]],
            dtype=np.float32,
        )
        mock_client.infer_raw.return_value = {"output0": end2end_output}
        mock_client_cls.return_value = mock_client

        result = run_folder_inference(
            image_dir=tmp_path,
            model_name="yolo26s",
            triton_url="localhost:8001",
            end2end=True,
        )

        assert len(result.results) == 1
        assert len(result.results[0].detections) == 1
        assert result.results[0].detections[0].confidence == pytest.approx(0.9)

    @patch("triton_model_tools.folder_inference._load_image")
    @patch("triton_model_tools.folder_inference.TritonInferenceClient")
    def test_custom_output_names(
        self,
        mock_client_cls: MagicMock,
        mock_load_image: MagicMock,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "img.jpg").write_bytes(b"fake")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_load_image.return_value = fake_image

        mock_client = MagicMock()
        mock_client.infer_raw.return_value = {
            "detections": np.zeros((1, 84, 8400), dtype=np.float32),
        }
        mock_client_cls.return_value = mock_client

        run_folder_inference(
            image_dir=tmp_path,
            model_name="yolo11s",
            triton_url="localhost:8001",
            output_names=["detections"],
        )

        call_args = mock_client.infer_raw.call_args
        assert call_args[0][2] == ["detections"]

    def test_raises_on_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No image files found"):
            run_folder_inference(
                image_dir=tmp_path,
                model_name="yolo11s",
            )
