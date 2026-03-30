"""Tests for Triton model repository directory builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from triton_model_tools.models import (
    TritonDataType,
    TritonModelConfig,
    TritonTensorSpec,
)
from triton_model_tools.repository import build_model_directory

if TYPE_CHECKING:
    from pathlib import Path


def _make_dummy_onnx(path: Path) -> Path:
    """Create a minimal dummy file to act as an ONNX model artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x08\x07" * 16)
    return path


def _make_test_config(name: str = "test_model") -> TritonModelConfig:
    return TritonModelConfig(
        name=name,
        inputs=[
            TritonTensorSpec(
                name="images",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1, 3, 640, 640],
            ),
        ],
        outputs=[
            TritonTensorSpec(
                name="output0",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1, 300, 6],
            ),
        ],
    )


class TestBuildModelDirectory:
    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        onnx_source = _make_dummy_onnx(tmp_path / "export" / "model.onnx")
        config = _make_test_config()

        model_dir = build_model_directory(repo_root, config, onnx_source)

        assert model_dir == repo_root / "test_model"
        assert model_dir.is_dir()
        assert (model_dir / "1").is_dir()

    def test_writes_config_pbtxt(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        onnx_source = _make_dummy_onnx(tmp_path / "export" / "model.onnx")
        config = _make_test_config()

        model_dir = build_model_directory(repo_root, config, onnx_source)
        config_path = model_dir / "config.pbtxt"

        assert config_path.exists()
        content = config_path.read_text()
        assert 'name: "test_model"' in content
        assert 'platform: "onnxruntime_onnx"' in content

    def test_copies_onnx_as_model_onnx(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        onnx_source = _make_dummy_onnx(tmp_path / "export" / "model.onnx")
        original_content = onnx_source.read_bytes()
        config = _make_test_config()

        model_dir = build_model_directory(repo_root, config, onnx_source)
        target_onnx = model_dir / "1" / "model.onnx"

        assert target_onnx.exists()
        assert target_onnx.read_bytes() == original_content

    def test_preserves_source_onnx(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        onnx_source = _make_dummy_onnx(tmp_path / "export" / "model.onnx")
        config = _make_test_config()

        build_model_directory(repo_root, config, onnx_source)

        assert onnx_source.exists()

    def test_existing_directory_is_overwritten(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        config = _make_test_config()

        onnx_v1 = _make_dummy_onnx(tmp_path / "v1" / "model.onnx")
        build_model_directory(repo_root, config, onnx_v1)

        onnx_v2 = tmp_path / "v2" / "model.onnx"
        onnx_v2.parent.mkdir(parents=True, exist_ok=True)
        onnx_v2.write_bytes(b"\xff" * 64)
        build_model_directory(repo_root, config, onnx_v2)

        target = repo_root / "test_model" / "1" / "model.onnx"
        assert target.read_bytes() == b"\xff" * 64

    def test_custom_model_name(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "model_repository"
        onnx_source = _make_dummy_onnx(tmp_path / "export" / "model.onnx")
        config = _make_test_config(name="yolo26s")

        model_dir = build_model_directory(repo_root, config, onnx_source)

        assert model_dir.name == "yolo26s"
        assert (model_dir / "config.pbtxt").exists()
        assert (model_dir / "1" / "model.onnx").exists()
