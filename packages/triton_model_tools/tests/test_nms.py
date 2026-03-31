"""Tests for NMS postprocessing model generation."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from triton_model_tools.export.nms import (
    _NMS_SCRIPT_PATH,
    NMS_MODEL_NAME,
    NmsModelConfig,
    build_nms_model_config,
    build_nms_model_directory,
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


class TestNmsScript:
    def test_script_file_exists(self) -> None:
        assert _NMS_SCRIPT_PATH.exists()

    def test_script_is_valid_python(self) -> None:
        ast.parse(_NMS_SCRIPT_PATH.read_text())

    def test_contains_nms_operations(self) -> None:
        script = _NMS_SCRIPT_PATH.read_text()
        assert "nms" in script.lower()
        assert "torchvision.ops.nms" in script

    def test_contains_triton_python_model_class(self) -> None:
        script = _NMS_SCRIPT_PATH.read_text()
        assert "class TritonPythonModel" in script

    def test_contains_initialize_and_execute(self) -> None:
        script = _NMS_SCRIPT_PATH.read_text()
        assert "def initialize" in script
        assert "def execute" in script

    def test_reads_parameters_from_model_config(self) -> None:
        script = _NMS_SCRIPT_PATH.read_text()
        assert 'params["confidence_threshold"]' in script
        assert 'params["iou_threshold"]' in script
        assert 'params["max_detections"]' in script

    def test_contains_numpy_fallback(self) -> None:
        script = _NMS_SCRIPT_PATH.read_text()
        assert "_numpy_nms" in script
        assert "ImportError" in script

    def test_contains_xywh_to_xyxy_conversion(self) -> None:
        script = _NMS_SCRIPT_PATH.read_text()
        assert "_xywh_to_xyxy" in script

    def test_uses_pb_utils(self) -> None:
        script = _NMS_SCRIPT_PATH.read_text()
        assert "pb_utils" in script
        assert "get_input_tensor_by_name" in script
        assert "InferenceResponse" in script

    def test_script_has_no_finalize(self) -> None:
        script = _NMS_SCRIPT_PATH.read_text()
        assert "finalize" not in script


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
        assert model_config.outputs[0].dims == [1, -1, 6]

    def test_output_dims_always_dynamic(self) -> None:
        config = NmsModelConfig(max_detections=100)
        model_config = build_nms_model_config(config)
        assert model_config.outputs[0].dims == [1, -1, 6]

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

    def test_parameters_contain_config_values(self) -> None:
        config = NmsModelConfig(confidence_threshold=0.3, iou_threshold=0.5, max_detections=200)
        model_config = build_nms_model_config(config)
        assert model_config.parameters["confidence_threshold"] == "0.3"
        assert model_config.parameters["iou_threshold"] == "0.5"
        assert model_config.parameters["max_detections"] == "200"

    def test_pbtxt_renders_parameters(self) -> None:
        config = NmsModelConfig()
        model_config = build_nms_model_config(config)
        pbtxt = model_config.to_pbtxt()
        assert 'key: "confidence_threshold"' in pbtxt
        assert 'key: "iou_threshold"' in pbtxt
        assert 'key: "max_detections"' in pbtxt


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

    def test_config_pbtxt_contains_parameters(self, tmp_path: Path) -> None:
        config = NmsModelConfig(confidence_threshold=0.5)
        build_nms_model_directory(tmp_path, config)
        content = (tmp_path / NMS_MODEL_NAME / "config.pbtxt").read_text()
        assert 'key: "confidence_threshold"' in content
        assert 'string_value: "0.5"' in content

    def test_returns_model_directory_path(self, tmp_path: Path) -> None:
        config = NmsModelConfig()
        result = build_nms_model_directory(tmp_path, config)
        assert result == tmp_path / NMS_MODEL_NAME
