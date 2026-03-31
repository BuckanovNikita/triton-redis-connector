"""Tests for DALI preprocessing pipeline and ensemble config generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from triton_model_tools.export.dali import (
    DALI_ENSEMBLE_SUFFIX,
    DALI_PREPROCESS_SUFFIX,
    DaliPipelineConfig,
    build_dali_model_config,
    build_dali_nms_ensemble_config,
    build_ensemble_config,
    build_ensemble_directory,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestBuildDaliModelConfig:
    def test_returns_dali_backend(self) -> None:
        config = DaliPipelineConfig()
        model_config = build_dali_model_config("yolo11s", config)
        assert model_config.backend == "dali"

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


class TestBuildEnsembleConfig:
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


class TestBuildEnsembleDirectory:
    def test_creates_preprocess_directory(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        build_ensemble_directory(tmp_path, "yolo11s", config)

        preprocess_dir = tmp_path / f"yolo11s{DALI_PREPROCESS_SUFFIX}"
        assert preprocess_dir.is_dir()
        assert (preprocess_dir / "1").is_dir()

    def test_creates_ensemble_directory(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        build_ensemble_directory(tmp_path, "yolo11s", config)

        ensemble_dir = tmp_path / f"yolo11s{DALI_ENSEMBLE_SUFFIX}"
        assert ensemble_dir.is_dir()
        assert (ensemble_dir / "1").is_dir()


class TestBuildDaliNmsEnsembleConfig:
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
