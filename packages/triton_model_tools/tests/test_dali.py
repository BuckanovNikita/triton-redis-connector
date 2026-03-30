"""Tests for DALI preprocessing pipeline and ensemble config generation."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from triton_model_tools.dali import (
    DALI_ENSEMBLE_SUFFIX,
    DALI_NMS_ENSEMBLE_SUFFIX,
    DALI_PREPROCESS_SUFFIX,
    DaliPipelineConfig,
    build_dali_model_config,
    build_dali_nms_ensemble_config,
    build_ensemble_config,
    build_ensemble_directory,
    generate_dali_pipeline_script,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestDaliPipelineConfig:
    def test_default_values(self) -> None:
        config = DaliPipelineConfig()
        assert config.target_size == (640, 640)
        assert config.pad_value == 114
        assert config.normalize_div == 255.0
        assert config.output_layout == "CHW"

    def test_custom_values(self) -> None:
        config = DaliPipelineConfig(
            target_size=(320, 320),
            pad_value=0,
            normalize_div=128.0,
            output_layout="HWC",
        )
        assert config.target_size == (320, 320)
        assert config.pad_value == 0
        assert config.normalize_div == 128.0
        assert config.output_layout == "HWC"


class TestGenerateDaliPipelineScript:
    def test_produces_valid_python(self) -> None:
        config = DaliPipelineConfig()
        script = generate_dali_pipeline_script(config)
        ast.parse(script)

    def test_contains_image_decoder(self) -> None:
        config = DaliPipelineConfig()
        script = generate_dali_pipeline_script(config)
        assert "fn.decoders.image" in script

    def test_contains_resize(self) -> None:
        config = DaliPipelineConfig()
        script = generate_dali_pipeline_script(config)
        assert "fn.resize" in script

    def test_contains_centered_letterbox_paste_with_114(self) -> None:
        config = DaliPipelineConfig()
        script = generate_dali_pipeline_script(config)
        assert "fn.paste" in script
        assert "(114, 114, 114)" in script

    def test_contains_centered_padding_offsets(self) -> None:
        config = DaliPipelineConfig()
        script = generate_dali_pipeline_script(config)
        assert "pad_top" in script
        assert "pad_left" in script

    def test_contains_crop_to_target_size(self) -> None:
        config = DaliPipelineConfig()
        script = generate_dali_pipeline_script(config)
        assert "fn.crop" in script

    def test_contains_bgr_to_rgb_conversion(self) -> None:
        config = DaliPipelineConfig()
        script = generate_dali_pipeline_script(config)
        assert "fn.color_space_conversion" in script
        assert "types.BGR" in script
        assert "types.RGB" in script

    def test_contains_normalize(self) -> None:
        config = DaliPipelineConfig()
        script = generate_dali_pipeline_script(config)
        assert "fn.crop_mirror_normalize" in script
        assert "255.0" in script

    def test_contains_chw_layout(self) -> None:
        config = DaliPipelineConfig()
        script = generate_dali_pipeline_script(config)
        assert 'output_layout="CHW"' in script

    def test_aspect_preserving_resize(self) -> None:
        config = DaliPipelineConfig()
        script = generate_dali_pipeline_script(config)
        assert "fn.min" in script
        assert "scale" in script

    def test_custom_pad_value_in_script(self) -> None:
        config = DaliPipelineConfig(pad_value=128)
        script = generate_dali_pipeline_script(config)
        assert "(128, 128, 128)" in script

    def test_custom_target_size_in_script(self) -> None:
        config = DaliPipelineConfig(target_size=(320, 320))
        script = generate_dali_pipeline_script(config)
        assert "target_h = 320" in script
        assert "target_w = 320" in script


class TestBuildDaliModelConfig:
    def test_returns_dali_backend(self) -> None:
        config = DaliPipelineConfig()
        model_config = build_dali_model_config("yolo11s", config)
        assert model_config.backend == "dali"

    def test_model_name_has_preprocess_suffix(self) -> None:
        config = DaliPipelineConfig()
        model_config = build_dali_model_config("yolo11s", config)
        assert model_config.name == f"yolo11s{DALI_PREPROCESS_SUFFIX}"

    def test_input_is_uint8_bytes(self) -> None:
        config = DaliPipelineConfig()
        model_config = build_dali_model_config("yolo11s", config)
        assert len(model_config.inputs) == 1
        assert model_config.inputs[0].data_type.value == "TYPE_UINT8"
        assert model_config.inputs[0].dims == [-1]

    def test_output_is_fp32_chw(self) -> None:
        config = DaliPipelineConfig()
        model_config = build_dali_model_config("yolo11s", config)
        assert len(model_config.outputs) == 1
        assert model_config.outputs[0].data_type.value == "TYPE_FP32"
        assert model_config.outputs[0].dims == [3, 640, 640]

    def test_custom_target_size_in_output_dims(self) -> None:
        config = DaliPipelineConfig(target_size=(320, 320))
        model_config = build_dali_model_config("yolo11s", config)
        assert model_config.outputs[0].dims == [3, 320, 320]

    def test_pbtxt_renders_backend(self) -> None:
        config = DaliPipelineConfig()
        model_config = build_dali_model_config("yolo11s", config)
        pbtxt = model_config.to_pbtxt()
        assert 'backend: "dali"' in pbtxt
        assert "platform:" not in pbtxt


class TestBuildEnsembleConfig:
    def test_returns_ensemble_platform(self) -> None:
        model_config = build_ensemble_config(
            model_name="yolo11s_dali_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
        )
        assert model_config.platform == "ensemble"

    def test_has_ensemble_scheduling(self) -> None:
        model_config = build_ensemble_config(
            model_name="yolo11s_dali_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
        )
        assert model_config.ensemble_scheduling is not None
        assert len(model_config.ensemble_scheduling) == 2

    def test_first_step_is_preprocess(self) -> None:
        model_config = build_ensemble_config(
            model_name="yolo11s_dali_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
        )
        assert model_config.ensemble_scheduling is not None
        first_step = model_config.ensemble_scheduling[0]
        assert first_step.model_name == "yolo11s_dali_preprocess"

    def test_second_step_is_inference(self) -> None:
        model_config = build_ensemble_config(
            model_name="yolo11s_dali_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
        )
        assert model_config.ensemble_scheduling is not None
        second_step = model_config.ensemble_scheduling[1]
        assert second_step.model_name == "yolo11s"

    def test_step_mappings_chain_correctly(self) -> None:
        model_config = build_ensemble_config(
            model_name="yolo11s_dali_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
        )
        assert model_config.ensemble_scheduling is not None
        preprocess_step = model_config.ensemble_scheduling[0]
        inference_step = model_config.ensemble_scheduling[1]

        preprocess_output_value = preprocess_step.output_map[0].value
        inference_input_value = inference_step.input_map[0].value
        assert preprocess_output_value == inference_input_value

    def test_pbtxt_renders_ensemble_scheduling(self) -> None:
        model_config = build_ensemble_config(
            model_name="yolo11s_dali_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
        )
        pbtxt = model_config.to_pbtxt()
        assert "ensemble_scheduling {" in pbtxt
        assert "instance_group" not in pbtxt
        assert 'platform: "ensemble"' in pbtxt

    def test_input_output_tensor_specs(self) -> None:
        model_config = build_ensemble_config(
            model_name="yolo11s_dali_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
        )
        assert len(model_config.inputs) == 1
        assert model_config.inputs[0].name == "INPUT_IMAGE"
        assert len(model_config.outputs) == 1
        assert model_config.outputs[0].name == "OUTPUT_DETECTIONS"


class TestBuildEnsembleDirectory:
    def test_creates_preprocess_directory(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        build_ensemble_directory(tmp_path, "yolo11s", config)

        preprocess_dir = tmp_path / f"yolo11s{DALI_PREPROCESS_SUFFIX}"
        assert preprocess_dir.is_dir()
        assert (preprocess_dir / "1").is_dir()

    def test_creates_dali_model_file(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        build_ensemble_directory(tmp_path, "yolo11s", config)

        dali_model = tmp_path / f"yolo11s{DALI_PREPROCESS_SUFFIX}" / "1" / "model.dali"
        assert dali_model.exists()
        content = dali_model.read_text()
        ast.parse(content)

    def test_creates_preprocess_config_pbtxt(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        build_ensemble_directory(tmp_path, "yolo11s", config)

        config_path = tmp_path / f"yolo11s{DALI_PREPROCESS_SUFFIX}" / "config.pbtxt"
        assert config_path.exists()
        content = config_path.read_text()
        assert 'backend: "dali"' in content

    def test_creates_ensemble_directory(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        build_ensemble_directory(tmp_path, "yolo11s", config)

        ensemble_dir = tmp_path / f"yolo11s{DALI_ENSEMBLE_SUFFIX}"
        assert ensemble_dir.is_dir()
        assert (ensemble_dir / "1").is_dir()

    def test_creates_ensemble_config_pbtxt(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        build_ensemble_directory(tmp_path, "yolo11s", config)

        config_path = tmp_path / f"yolo11s{DALI_ENSEMBLE_SUFFIX}" / "config.pbtxt"
        assert config_path.exists()
        content = config_path.read_text()
        assert 'platform: "ensemble"' in content
        assert "ensemble_scheduling" in content

    def test_returns_ensemble_directory_path(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        result = build_ensemble_directory(tmp_path, "yolo11s", config)
        assert result == tmp_path / f"yolo11s{DALI_ENSEMBLE_SUFFIX}"

    def test_ensemble_version_dir_is_empty(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        build_ensemble_directory(tmp_path, "yolo11s", config)

        version_dir = tmp_path / f"yolo11s{DALI_ENSEMBLE_SUFFIX}" / "1"
        assert version_dir.is_dir()
        children = list(version_dir.iterdir())
        assert len(children) == 0

    def test_creates_nms_ensemble_when_nms_provided(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        result = build_ensemble_directory(
            tmp_path,
            "yolo11s",
            config,
            nms_model_name="nms_postprocess",
        )
        assert result == tmp_path / f"yolo11s{DALI_NMS_ENSEMBLE_SUFFIX}"
        config_path = result / "config.pbtxt"
        assert config_path.exists()
        content = config_path.read_text()
        assert 'model_name: "nms_postprocess"' in content
        assert "ensemble_scheduling" in content

    def test_creates_standard_ensemble_when_nms_absent(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        result = build_ensemble_directory(tmp_path, "yolo11s", config)
        assert result == tmp_path / f"yolo11s{DALI_ENSEMBLE_SUFFIX}"
        config_path = result / "config.pbtxt"
        content = config_path.read_text()
        assert "nms_postprocess" not in content


class TestBuildDaliNmsEnsembleConfig:
    def test_returns_ensemble_platform(self) -> None:
        model_config = build_dali_nms_ensemble_config(
            model_name="yolo11s_dali_nms_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
            nms_name="nms_postprocess",
        )
        assert model_config.platform == "ensemble"

    def test_has_three_steps(self) -> None:
        model_config = build_dali_nms_ensemble_config(
            model_name="yolo11s_dali_nms_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
            nms_name="nms_postprocess",
        )
        assert model_config.ensemble_scheduling is not None
        assert len(model_config.ensemble_scheduling) == 3

    def test_steps_chain_correctly(self) -> None:
        model_config = build_dali_nms_ensemble_config(
            model_name="yolo11s_dali_nms_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
            nms_name="nms_postprocess",
        )
        assert model_config.ensemble_scheduling is not None
        preprocess_step = model_config.ensemble_scheduling[0]
        inference_step = model_config.ensemble_scheduling[1]
        nms_step = model_config.ensemble_scheduling[2]

        assert preprocess_step.model_name == "yolo11s_dali_preprocess"
        assert inference_step.model_name == "yolo11s"
        assert nms_step.model_name == "nms_postprocess"

        assert preprocess_step.output_map[0].value == inference_step.input_map[0].value
        assert inference_step.output_map[0].value == nms_step.input_map[0].value

    def test_input_output_tensor_specs(self) -> None:
        model_config = build_dali_nms_ensemble_config(
            model_name="yolo11s_dali_nms_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
            nms_name="nms_postprocess",
        )
        assert len(model_config.inputs) == 1
        assert model_config.inputs[0].name == "INPUT_IMAGE"
        assert len(model_config.outputs) == 1
        assert model_config.outputs[0].name == "OUTPUT_DETECTIONS"
