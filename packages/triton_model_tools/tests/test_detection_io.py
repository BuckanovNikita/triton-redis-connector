"""Tests for detection I/O utilities (txt format reading/writing)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from triton_model_tools.inference.detection_io import (
    load_detections_from_txt,
    load_folder_detections,
    save_detections_to_txt,
)
from triton_model_tools.models.detection import Detection


def _sample_detections() -> list[Detection]:
    return [
        Detection(class_id=0, x1=10.5, y1=20.3, x2=100.0, y2=200.0, confidence=0.95),
        Detection(class_id=1, x1=50.0, y1=60.0, x2=150.0, y2=250.0, confidence=0.80),
        Detection(class_id=2, x1=0.0, y1=0.0, x2=640.0, y2=480.0, confidence=0.50),
    ]


class TestSaveDetectionsToTxt:
    def test_writes_correct_format(self, tmp_path: Path) -> None:
        detections = _sample_detections()
        output_path = tmp_path / "test.txt"
        save_detections_to_txt(detections, output_path)

        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == 3
        parts = lines[0].split()
        assert parts[0] == "0"
        assert float(parts[1]) == pytest.approx(10.5)
        assert float(parts[5]) == pytest.approx(0.95)

    def test_empty_detections(self, tmp_path: Path) -> None:
        output_path = tmp_path / "empty.txt"
        save_detections_to_txt([], output_path)
        assert output_path.read_text() == ""

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        output_path = tmp_path / "sub" / "dir" / "test.txt"
        save_detections_to_txt(_sample_detections(), output_path)
        assert output_path.exists()


class TestLoadDetectionsFromTxt:
    def test_roundtrip(self, tmp_path: Path) -> None:
        detections = _sample_detections()
        txt_path = tmp_path / "test.txt"
        save_detections_to_txt(detections, txt_path)

        result = load_detections_from_txt(txt_path, image_path="test_img", model_name="yolo11s")
        assert result.image_path == "test_img"
        assert result.model_name == "yolo11s"
        assert len(result.detections) == 3
        assert result.detections[0].class_id == 0
        assert result.detections[0].x1 == pytest.approx(10.5, abs=1e-4)
        assert result.detections[0].confidence == pytest.approx(0.95, abs=1e-4)

    def test_empty_file(self, tmp_path: Path) -> None:
        txt_path = tmp_path / "empty.txt"
        txt_path.write_text("")

        result = load_detections_from_txt(txt_path, image_path="empty", model_name="model")
        assert len(result.detections) == 0

    def test_malformed_line_skipped(self, tmp_path: Path) -> None:
        txt_path = tmp_path / "bad.txt"
        txt_path.write_text("0 10.0 20.0\n1 50.0 60.0 150.0 250.0 0.80\n")

        result = load_detections_from_txt(txt_path, image_path="bad", model_name="model")
        assert len(result.detections) == 1
        assert result.detections[0].class_id == 1


class TestLoadFolderDetections:
    def test_loads_all_txt_files(self, tmp_path: Path) -> None:
        for i in range(3):
            detections = [Detection(class_id=i, x1=0, y1=0, x2=100, y2=100, confidence=0.9)]
            save_detections_to_txt(detections, tmp_path / f"image_{i:04d}.txt")

        results = load_folder_detections(tmp_path, model_name="yolo11s")
        assert len(results) == 3
        assert results[0].image_path == "image_0000"
        assert results[0].detections[0].class_id == 0

    def test_raises_on_empty_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No txt files found"):
            load_folder_detections(tmp_path, model_name="yolo11s")
