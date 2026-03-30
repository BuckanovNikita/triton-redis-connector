"""Tests for NMS postprocessing model generation."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from triton_model_tools.nms import (
    NMS_MODEL_NAME,
    NmsModelConfig,
    build_nms_model_config,
    build_nms_model_directory,
    generate_nms_model_script,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestNmsModelConfig:
    def test_default_values(self) -> None:
        config = NmsModelConfig()
        assert config.confidence_threshold == 0.25
        assert config.iou_threshold == 0.45
        assert config.max_detections == 300

    def test_custom_values(self) -> None:
        config = NmsModelConfig(
            confidence_threshold=0.5,
            iou_threshold=0.6,
            max_detections=100,
        )
        assert config.confidence_threshold == 0.5
        assert config.iou_threshold == 0.6
        assert config.max_detections == 100

    def test_confidence_threshold_bounds(self) -> None:
        config_min = NmsModelConfig(confidence_threshold=0.0)
        assert config_min.confidence_threshold == 0.0

        config_max = NmsModelConfig(confidence_threshold=1.0)
        assert config_max.confidence_threshold == 1.0

    def test_iou_threshold_bounds(self) -> None:
        config_min = NmsModelConfig(iou_threshold=0.0)
        assert config_min.iou_threshold == 0.0

        config_max = NmsModelConfig(iou_threshold=1.0)
        assert config_max.iou_threshold == 1.0


class TestGenerateNmsModelScript:
    def test_produces_valid_python(self) -> None:
        config = NmsModelConfig()
        script = generate_nms_model_script(config)
        ast.parse(script)

    def test_contains_nms_operations(self) -> None:
        config = NmsModelConfig()
        script = generate_nms_model_script(config)
        assert "nms" in script.lower()
        assert "torchvision.ops.nms" in script

    def test_contains_triton_python_model_class(self) -> None:
        config = NmsModelConfig()
        script = generate_nms_model_script(config)
        assert "class TritonPythonModel" in script

    def test_contains_initialize_and_execute(self) -> None:
        config = NmsModelConfig()
        script = generate_nms_model_script(config)
        assert "def initialize" in script
        assert "def execute" in script

    def test_contains_confidence_threshold(self) -> None:
        config = NmsModelConfig(confidence_threshold=0.3)
        script = generate_nms_model_script(config)
        assert "0.3" in script

    def test_contains_iou_threshold(self) -> None:
        config = NmsModelConfig(iou_threshold=0.5)
        script = generate_nms_model_script(config)
        assert "0.5" in script

    def test_contains_max_detections(self) -> None:
        config = NmsModelConfig(max_detections=200)
        script = generate_nms_model_script(config)
        assert "200" in script

    def test_contains_numpy_fallback(self) -> None:
        config = NmsModelConfig()
        script = generate_nms_model_script(config)
        assert "_numpy_nms" in script
        assert "ImportError" in script

    def test_contains_xywh_to_xyxy_conversion(self) -> None:
        config = NmsModelConfig()
        script = generate_nms_model_script(config)
        assert "_xywh_to_xyxy" in script

    def test_uses_pb_utils(self) -> None:
        config = NmsModelConfig()
        script = generate_nms_model_script(config)
        assert "pb_utils" in script
        assert "get_input_tensor_by_name" in script
        assert "InferenceResponse" in script


class TestBuildNmsModelConfig:
    def test_returns_python_backend(self) -> None:
        config = NmsModelConfig()
        model_config = build_nms_model_config(config)
        assert model_config.backend == "python"

    def test_model_name_is_nms_postprocess(self) -> None:
        config = NmsModelConfig()
        model_config = build_nms_model_config(config)
        assert model_config.name == NMS_MODEL_NAME

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
        assert model_config.outputs[0].dims == [1, 300, 6]

    def test_output_dims_reflect_max_detections(self) -> None:
        config = NmsModelConfig(max_detections=100)
        model_config = build_nms_model_config(config)
        assert model_config.outputs[0].dims == [1, 100, 6]

    def test_runs_on_cpu(self) -> None:
        config = NmsModelConfig()
        model_config = build_nms_model_config(config)
        assert model_config.instance_kind.value == "KIND_CPU"

    def test_pbtxt_renders_backend(self) -> None:
        config = NmsModelConfig()
        model_config = build_nms_model_config(config)
        pbtxt = model_config.to_pbtxt()
        assert 'backend: "python"' in pbtxt
        assert "platform:" not in pbtxt


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

    def test_creates_config_pbtxt(self, tmp_path: Path) -> None:
        config = NmsModelConfig()
        build_nms_model_directory(tmp_path, config)
        config_path = tmp_path / NMS_MODEL_NAME / "config.pbtxt"
        assert config_path.exists()
        content = config_path.read_text()
        assert 'backend: "python"' in content
        assert f'name: "{NMS_MODEL_NAME}"' in content

    def test_returns_model_directory_path(self, tmp_path: Path) -> None:
        config = NmsModelConfig()
        result = build_nms_model_directory(tmp_path, config)
        assert result == tmp_path / NMS_MODEL_NAME
