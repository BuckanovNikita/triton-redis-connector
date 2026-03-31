"""Tests for NMS postprocessing model generation."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from triton_model_tools.export.nms import (
    NMS_MODEL_NAME,
    NmsModelConfig,
    build_nms_model_config,
    build_nms_model_directory,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestBuildNmsModelConfig:
    def test_returns_python_backend(self) -> None:
        config = NmsModelConfig()
        model_config = build_nms_model_config(config)
        assert model_config.backend == "python"

    def test_input_is_fp32_dynamic(self) -> None:
        config = NmsModelConfig()
        model_config = build_nms_model_config(config)
        assert len(model_config.inputs) == 1
        assert model_config.inputs[0].data_type.value == "TYPE_FP32"
        assert model_config.inputs[0].name == "INPUT_0"

    def test_output_is_fp32_detections(self) -> None:
        config = NmsModelConfig()
        model_config = build_nms_model_config(config)
        assert len(model_config.outputs) == 1
        assert model_config.outputs[0].data_type.value == "TYPE_FP32"
        assert model_config.outputs[0].name == "OUTPUT_0"
        assert model_config.outputs[0].dims == [1, -1, 6]

    def test_output_dims_always_dynamic(self) -> None:
        config = NmsModelConfig(max_detections=100)
        model_config = build_nms_model_config(config)
        assert model_config.outputs[0].dims == [1, -1, 6]

    def test_parameters_contain_config_values(self) -> None:
        config = NmsModelConfig(confidence_threshold=0.3, iou_threshold=0.5, max_detections=200)
        model_config = build_nms_model_config(config)
        assert model_config.parameters["confidence_threshold"] == "0.3"
        assert model_config.parameters["iou_threshold"] == "0.5"
        assert model_config.parameters["max_detections"] == "200"


class TestBuildNmsModelDirectory:
    def test_creates_model_directory(self, tmp_path: Path) -> None:
        config = NmsModelConfig()
        model_dir = build_nms_model_directory(tmp_path, config)
        assert model_dir == tmp_path / NMS_MODEL_NAME
        assert model_dir.is_dir()

    def test_creates_version_directory(self, tmp_path: Path) -> None:
        config = NmsModelConfig()
        build_nms_model_directory(tmp_path, config)
        version_dir = tmp_path / NMS_MODEL_NAME / "1"
        assert version_dir.is_dir()

    def test_creates_model_py(self, tmp_path: Path) -> None:
        config = NmsModelConfig()
        build_nms_model_directory(tmp_path, config)
        model_py = tmp_path / NMS_MODEL_NAME / "1" / "model.py"
        assert model_py.exists()
        content = model_py.read_text()
        ast.parse(content)
