"""Tests for YOLO model export to ONNX and TensorRT with Triton repository building."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from triton_model_tools.export.exporter import (
    TENSORRT_NAME_SUFFIX,
    _export_onnx,
    _export_tensorrt,
    export_all_models,
    export_model,
)
from triton_model_tools.models import (
    EXPORT_FORMAT_TO_FILENAME,
    ExportFormat,
    YoloModelVariant,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_variant(
    name: str = "yolo11s",
    imgsz: int | tuple[int, int] = 640,
    *,
    end2end_export: bool = False,
) -> YoloModelVariant:
    return YoloModelVariant(
        name=name,
        ultralytics_model=f"{name}.pt",
        imgsz=imgsz,
        end2end_export=end2end_export,
    )


def _make_mock_yolo(export_dir: Path, suffix: str) -> MagicMock:
    mock_yolo_cls = MagicMock()
    mock_yolo_instance = MagicMock()
    mock_yolo_cls.return_value = mock_yolo_instance

    exported_file = export_dir / f"model{suffix}"
    exported_file.parent.mkdir(parents=True, exist_ok=True)
    exported_file.write_bytes(b"\x00" * 64)

    mock_yolo_instance.export.return_value = str(exported_file)
    return mock_yolo_cls


class TestExportOnnx:
    def test_creates_model_directory(self, tmp_path: Path) -> None:
        variant = _make_variant()
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        export_dir = tmp_path / "exports"
        mock_yolo_cls = _make_mock_yolo(export_dir, ".onnx")

        with (
            patch("triton_model_tools.export.exporter.YOLO", mock_yolo_cls),
            patch("triton_model_tools.export.exporter.inspect_onnx_model") as mock_inspect,
        ):
            from triton_model_tools.models import TritonDataType, TritonTensorSpec

            mock_inspect.return_value = (
                [TritonTensorSpec(name="images", data_type=TritonDataType.TYPE_FP32, dims=[1, 3, 640, 640])],
                [TritonTensorSpec(name="output0", data_type=TritonDataType.TYPE_FP32, dims=[1, 300, 6])],
            )
            model_dir = _export_onnx(variant, repo_root)

        assert model_dir == repo_root / "yolo11s"
        assert (model_dir / "config.pbtxt").exists()
        assert (model_dir / "1" / EXPORT_FORMAT_TO_FILENAME[ExportFormat.ONNX]).exists()

    def test_passes_half_to_yolo_export(self, tmp_path: Path) -> None:
        variant = _make_variant()
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        export_dir = tmp_path / "exports"
        mock_yolo_cls = _make_mock_yolo(export_dir, ".onnx")

        with (
            patch("triton_model_tools.export.exporter.YOLO", mock_yolo_cls),
            patch("triton_model_tools.export.exporter.inspect_onnx_model") as mock_inspect,
        ):
            from triton_model_tools.models import TritonDataType, TritonTensorSpec

            mock_inspect.return_value = (
                [TritonTensorSpec(name="images", data_type=TritonDataType.TYPE_FP16, dims=[1, 3, 640, 640])],
                [TritonTensorSpec(name="output0", data_type=TritonDataType.TYPE_FP16, dims=[1, 300, 6])],
            )
            _export_onnx(variant, repo_root, half=True)

        mock_instance = mock_yolo_cls.return_value
        export_call_kwargs: dict[str, Any] = mock_instance.export.call_args.kwargs
        assert export_call_kwargs["half"] is True

    def test_cleans_up_intermediate_onnx(self, tmp_path: Path) -> None:
        variant = _make_variant()
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        export_dir = tmp_path / "exports"
        mock_yolo_cls = _make_mock_yolo(export_dir, ".onnx")
        exported_file = export_dir / "model.onnx"

        with (
            patch("triton_model_tools.export.exporter.YOLO", mock_yolo_cls),
            patch("triton_model_tools.export.exporter.inspect_onnx_model") as mock_inspect,
        ):
            from triton_model_tools.models import TritonDataType, TritonTensorSpec

            mock_inspect.return_value = (
                [TritonTensorSpec(name="images", data_type=TritonDataType.TYPE_FP32, dims=[1, 3, 640, 640])],
                [TritonTensorSpec(name="output0", data_type=TritonDataType.TYPE_FP32, dims=[1, 300, 6])],
            )
            _export_onnx(variant, repo_root)

        assert not exported_file.exists()


class TestExportTensorrt:
    def test_creates_model_directory_with_trt_suffix(self, tmp_path: Path) -> None:
        variant = _make_variant()
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        export_dir = tmp_path / "exports"
        mock_yolo_cls = _make_mock_yolo(export_dir, ".engine")

        with patch("triton_model_tools.export.exporter.YOLO", mock_yolo_cls):
            model_dir = _export_tensorrt(variant, repo_root)

        expected_name = f"yolo11s{TENSORRT_NAME_SUFFIX}"
        assert model_dir == repo_root / expected_name
        assert model_dir.is_dir()

    def test_model_file_is_plan(self, tmp_path: Path) -> None:
        variant = _make_variant()
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        export_dir = tmp_path / "exports"
        mock_yolo_cls = _make_mock_yolo(export_dir, ".engine")

        with patch("triton_model_tools.export.exporter.YOLO", mock_yolo_cls):
            model_dir = _export_tensorrt(variant, repo_root)

        plan_file = model_dir / "1" / EXPORT_FORMAT_TO_FILENAME[ExportFormat.TENSORRT]
        assert plan_file.exists()

    def test_export_calls_yolo_with_engine_format_and_half(self, tmp_path: Path) -> None:
        variant = _make_variant()
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        export_dir = tmp_path / "exports"
        mock_yolo_cls = _make_mock_yolo(export_dir, ".engine")

        with patch("triton_model_tools.export.exporter.YOLO", mock_yolo_cls):
            _export_tensorrt(variant, repo_root)

        mock_instance = mock_yolo_cls.return_value
        export_call_kwargs: dict[str, Any] = mock_instance.export.call_args.kwargs
        assert export_call_kwargs["format"] == "engine"
        assert export_call_kwargs["half"] is True
        assert export_call_kwargs["imgsz"] == [640, 640]

    def test_half_false_uses_fp32_types(self, tmp_path: Path) -> None:
        variant = _make_variant()
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        export_dir = tmp_path / "exports"
        mock_yolo_cls = _make_mock_yolo(export_dir, ".engine")

        with patch("triton_model_tools.export.exporter.YOLO", mock_yolo_cls):
            model_dir = _export_tensorrt(variant, repo_root, half=False)

        config_text = (model_dir / "config.pbtxt").read_text()
        assert "TYPE_FP32" in config_text
        assert "TYPE_FP16" not in config_text

        mock_instance = mock_yolo_cls.return_value
        export_call_kwargs: dict[str, Any] = mock_instance.export.call_args.kwargs
        assert export_call_kwargs["half"] is False

    def test_end2end_variant_has_correct_output_dims(self, tmp_path: Path) -> None:
        variant = _make_variant(end2end_export=True)
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        export_dir = tmp_path / "exports"
        mock_yolo_cls = _make_mock_yolo(export_dir, ".engine")

        with patch("triton_model_tools.export.exporter.YOLO", mock_yolo_cls):
            model_dir = _export_tensorrt(variant, repo_root)

        config_text = (model_dir / "config.pbtxt").read_text()
        assert "1, -1, 6" in config_text

    def test_non_end2end_variant_has_default_output_dims(self, tmp_path: Path) -> None:
        variant = _make_variant(end2end_export=False)
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        export_dir = tmp_path / "exports"
        mock_yolo_cls = _make_mock_yolo(export_dir, ".engine")

        with patch("triton_model_tools.export.exporter.YOLO", mock_yolo_cls):
            model_dir = _export_tensorrt(variant, repo_root)

        config_text = (model_dir / "config.pbtxt").read_text()
        assert "-1, -1" in config_text

    def test_cleans_up_intermediate_engine_file(self, tmp_path: Path) -> None:
        variant = _make_variant()
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        export_dir = tmp_path / "exports"
        mock_yolo_cls = _make_mock_yolo(export_dir, ".engine")
        engine_file = export_dir / "model.engine"

        with patch("triton_model_tools.export.exporter.YOLO", mock_yolo_cls):
            _export_tensorrt(variant, repo_root)

        assert not engine_file.exists()


class TestExportModel:
    def _run_export_model(
        self,
        tmp_path: Path,
        fmt: str,
    ) -> list[Path]:
        variant = _make_variant()
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        export_dir = tmp_path / "exports"

        mock_yolo_cls_onnx = _make_mock_yolo(export_dir / "onnx", ".onnx")
        mock_yolo_cls_trt = _make_mock_yolo(export_dir / "trt", ".engine")

        call_count = 0

        def yolo_side_effect(_model_path: str) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1 and fmt in ("onnx", "all"):
                return mock_yolo_cls_onnx.return_value
            return mock_yolo_cls_trt.return_value

        mock_yolo = MagicMock(side_effect=yolo_side_effect)

        with (
            patch("triton_model_tools.export.exporter.YOLO", mock_yolo),
            patch("triton_model_tools.export.exporter.inspect_onnx_model") as mock_inspect,
        ):
            from triton_model_tools.models import TritonDataType, TritonTensorSpec

            mock_inspect.return_value = (
                [TritonTensorSpec(name="images", data_type=TritonDataType.TYPE_FP32, dims=[1, 3, 640, 640])],
                [TritonTensorSpec(name="output0", data_type=TritonDataType.TYPE_FP32, dims=[1, 300, 6])],
            )
            return export_model(variant, repo_root, fmt=fmt)  # type: ignore[arg-type]

    def test_fmt_onnx_returns_single_path(self, tmp_path: Path) -> None:
        results = self._run_export_model(tmp_path, "onnx")
        assert len(results) == 1
        assert results[0].name == "yolo11s"

    def test_fmt_tensorrt_returns_single_path(self, tmp_path: Path) -> None:
        results = self._run_export_model(tmp_path, "tensorrt")
        assert len(results) == 1
        assert results[0].name == f"yolo11s{TENSORRT_NAME_SUFFIX}"

    def test_fmt_all_returns_two_paths(self, tmp_path: Path) -> None:
        results = self._run_export_model(tmp_path, "all")
        assert len(results) == 2
        names = {r.name for r in results}
        assert "yolo11s" in names
        assert f"yolo11s{TENSORRT_NAME_SUFFIX}" in names


class TestExportAllModels:
    def test_passes_format_to_export_model(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"

        with patch("triton_model_tools.export.exporter.export_model") as mock_export_model:
            mock_export_model.return_value = [repo_root / "yolo11s_trt"]
            results = export_all_models(
                repository_root=repo_root,
                model_names=["yolo11s"],
                fmt="tensorrt",
            )

        assert mock_export_model.call_count == 1
        call_kwargs = mock_export_model.call_args.kwargs
        assert call_kwargs["fmt"] == "tensorrt"
        assert len(results) == 1

    def test_fmt_all_collects_all_results(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"

        with patch("triton_model_tools.export.exporter.export_model") as mock_export_model:
            mock_export_model.return_value = [
                repo_root / "yolo11s",
                repo_root / f"yolo11s{TENSORRT_NAME_SUFFIX}",
            ]
            results = export_all_models(
                repository_root=repo_root,
                model_names=["yolo11s"],
                fmt="all",
            )

        assert len(results) == 2

    def test_default_fmt_is_onnx(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"

        with patch("triton_model_tools.export.exporter.export_model") as mock_export_model:
            mock_export_model.return_value = [repo_root / "yolo11s"]
            export_all_models(
                repository_root=repo_root,
                model_names=["yolo11s"],
            )

        call_kwargs = mock_export_model.call_args.kwargs
        assert call_kwargs["fmt"] == "onnx"

    def test_passes_half_to_export_model(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"

        with patch("triton_model_tools.export.exporter.export_model") as mock_export_model:
            mock_export_model.return_value = [repo_root / "yolo11s"]
            export_all_models(
                repository_root=repo_root,
                model_names=["yolo11s"],
                half=True,
            )

        call_kwargs = mock_export_model.call_args.kwargs
        assert call_kwargs["half"] is True

    def test_unknown_model_name_raises_value_error(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"

        with pytest.raises(ValueError, match="Unknown model names"):
            export_all_models(
                repository_root=repo_root,
                model_names=["nonexistent_model"],
            )
