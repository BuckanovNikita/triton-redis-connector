"""Tests for Triton model repository directory builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from triton_model_tools.export.repository import build_model_directory
from triton_model_tools.models import (
    TritonDataType,
    TritonModelConfig,
    TritonTensorSpec,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_dummy_model(path: Path) -> Path:
    """Create a minimal dummy file to act as a model artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x08\x07" * 16)
    return path


def _make_test_config(name: str = "test_model", **kwargs: object) -> TritonModelConfig:
    defaults: dict[str, object] = {
        "name": name,
        "inputs": [
            TritonTensorSpec(
                name="images",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1, 3, 640, 640],
            ),
        ],
        "outputs": [
            TritonTensorSpec(
                name="output0",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1, 300, 6],
            ),
        ],
    }
    defaults.update(kwargs)
    return TritonModelConfig(**defaults)  # type: ignore[arg-type]


class TestBuildModelDirectory:
    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        model_source = _make_dummy_model(tmp_path / "export" / "model.onnx")
        config = _make_test_config()

        model_dir = build_model_directory(repo_root, config, model_source=model_source)

        assert model_dir == repo_root / "test_model"
        assert model_dir.is_dir()
        assert (model_dir / "1").is_dir()

    def test_writes_config_pbtxt(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        model_source = _make_dummy_model(tmp_path / "export" / "model.onnx")
        config = _make_test_config()

        model_dir = build_model_directory(repo_root, config, model_source=model_source)
        config_path = model_dir / "config.pbtxt"

        assert config_path.exists()
        content = config_path.read_text()
        assert 'name: "test_model"' in content
        assert 'platform: "onnxruntime_onnx"' in content

    def test_copies_onnx_model(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        model_source = _make_dummy_model(tmp_path / "export" / "model.onnx")
        original_content = model_source.read_bytes()
        config = _make_test_config()

        model_dir = build_model_directory(repo_root, config, model_source=model_source)
        target_onnx = model_dir / "1" / "model.onnx"

        assert target_onnx.exists()
        assert target_onnx.read_bytes() == original_content

    def test_preserves_source_file(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        model_source = _make_dummy_model(tmp_path / "export" / "model.onnx")
        config = _make_test_config()

        build_model_directory(repo_root, config, model_source=model_source)

        assert model_source.exists()

    def test_existing_directory_is_overwritten(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        config = _make_test_config()

        model_v1 = _make_dummy_model(tmp_path / "v1" / "model.onnx")
        build_model_directory(repo_root, config, model_source=model_v1)

        model_v2 = tmp_path / "v2" / "model.onnx"
        model_v2.parent.mkdir(parents=True, exist_ok=True)
        model_v2.write_bytes(b"\xff" * 64)
        build_model_directory(repo_root, config, model_source=model_v2)

        target = repo_root / "test_model" / "1" / "model.onnx"
        assert target.read_bytes() == b"\xff" * 64

    def test_custom_model_name(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        model_source = _make_dummy_model(tmp_path / "export" / "model.onnx")
        config = _make_test_config(name="yolo26s")

        model_dir = build_model_directory(repo_root, config, model_source=model_source)

        assert model_dir.name == "yolo26s"
        assert (model_dir / "config.pbtxt").exists()
        assert (model_dir / "1" / "model.onnx").exists()


class TestBuildModelDirectoryTensorRT:
    def test_copies_plan_file(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        model_source = _make_dummy_model(tmp_path / "export" / "model.plan")
        config = _make_test_config(name="yolo11s_trt", platform="tensorrt_plan")

        model_dir = build_model_directory(repo_root, config, model_source=model_source)
        target_plan = model_dir / "1" / "model.plan"

        assert target_plan.exists()
        assert target_plan.read_bytes() == model_source.read_bytes()

    def test_plan_extension_derives_filename(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        model_source = _make_dummy_model(tmp_path / "export" / "yolo11s.plan")
        config = _make_test_config(name="yolo11s_trt", platform="tensorrt_plan")

        model_dir = build_model_directory(repo_root, config, model_source=model_source)

        assert (model_dir / "1" / "model.plan").exists()

    def test_explicit_model_filename(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        model_source = _make_dummy_model(tmp_path / "export" / "custom_model.engine")
        config = _make_test_config(name="custom_model", platform="tensorrt_plan")

        model_dir = build_model_directory(
            repo_root,
            config,
            model_source=model_source,
            model_filename="model.plan",
        )

        assert (model_dir / "1" / "model.plan").exists()
        assert not (model_dir / "1" / "custom_model.engine").exists()

    def test_unknown_extension_uses_source_name(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        model_source = _make_dummy_model(tmp_path / "export" / "model.bin")
        config = _make_test_config(name="custom")

        model_dir = build_model_directory(repo_root, config, model_source=model_source)

        assert (model_dir / "1" / "model.bin").exists()

    def test_model_filename_overrides_extension_detection(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        model_source = _make_dummy_model(tmp_path / "export" / "model.onnx")
        config = _make_test_config()

        model_dir = build_model_directory(
            repo_root,
            config,
            model_source=model_source,
            model_filename="custom_name.onnx",
        )

        assert (model_dir / "1" / "custom_name.onnx").exists()
        assert not (model_dir / "1" / "model.onnx").exists()
