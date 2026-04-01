"""Tests for DALI preprocessing pipeline and ensemble config generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest
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

try:
    import nvidia.dali  # noqa: F401

    _HAS_DALI = True
except ImportError:
    _HAS_DALI = False

requires_dali = pytest.mark.skipif(not _HAS_DALI, reason="nvidia-dali not installed")


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
        assert model_config.outputs[0].dims == [1, 3, 640, 640]

    def test_custom_target_size_in_output_dims(self) -> None:
        config = DaliPipelineConfig(target_size=(320, 320))
        model_config = build_dali_model_config("yolo11s", config)
        assert model_config.outputs[0].dims == [1, 3, 320, 320]


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


@requires_dali
class TestDaliMatchesUltralytics:
    """Verify DALI preprocessing produces results matching LetterBoxPreprocessor.

    JPEG encode/decode introduces lossy artifacts, so pixel values differ
    slightly between DALI (decodes JPEG internally) and OpenCV (works on raw
    pixels). Tests use tolerances that account for this while still catching
    any algorithmic mismatches (wrong resize, bad padding, channel swap errors).
    """

    @staticmethod
    def _preprocess_both(
        bgr_image: npt.NDArray[np.uint8],
        target_size: tuple[int, int],
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        from triton_model_tools.export.dali import run_dali_preprocess
        from triton_model_tools.inference import YoloPreprocessor

        config = DaliPipelineConfig(target_size=target_size)
        dali_out = run_dali_preprocess(bgr_image, config)

        preprocessor = YoloPreprocessor(target_size=target_size)
        cpu_out, _, _ = preprocessor.preprocess(bgr_image)
        return dali_out, cpu_out

    def test_output_shape_matches_square(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        dali_out, cpu_out = self._preprocess_both(image, (640, 640))
        assert dali_out.shape == cpu_out.shape == (1, 3, 640, 640)

    def test_output_shape_matches_nonsquare(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        dali_out, cpu_out = self._preprocess_both(image, (960, 640))
        assert dali_out.shape == cpu_out.shape == (1, 3, 960, 640)

    def test_values_close_landscape(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        dali_out, cpu_out = self._preprocess_both(image, (640, 640))
        np.testing.assert_allclose(dali_out, cpu_out, atol=0.02, rtol=0)

    def test_values_close_portrait(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (800, 400, 3), dtype=np.uint8)
        dali_out, cpu_out = self._preprocess_both(image, (640, 640))
        np.testing.assert_allclose(dali_out, cpu_out, atol=0.02, rtol=0)

    def test_values_close_nonsquare_target(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
        dali_out, cpu_out = self._preprocess_both(image, (384, 640))
        np.testing.assert_allclose(dali_out, cpu_out, atol=0.02, rtol=0)

    def test_padding_region_matches(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        dali_out, cpu_out = self._preprocess_both(image, (640, 640))

        scale = min(640 / 480, 640 / 640)
        new_h = int(480 * scale)
        pad_top = (640 - new_h) // 2

        if pad_top > 0:
            dali_pad = dali_out[0, :, :pad_top, :]
            cpu_pad = cpu_out[0, :, :pad_top, :]
            np.testing.assert_allclose(dali_pad, cpu_pad, atol=1e-6)

    def test_normalization_range(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        dali_out, cpu_out = self._preprocess_both(image, (640, 640))
        assert dali_out.min() >= 0.0
        assert dali_out.max() <= 1.0
        assert cpu_out.min() >= 0.0
        assert cpu_out.max() <= 1.0

    def test_bgr_to_rgb_channel_order(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 200
        image[:, :, 1] = 0
        image[:, :, 2] = 100

        dali_out, cpu_out = self._preprocess_both(image, (64, 64))

        dali_r_mean = float(dali_out[0, 0].mean())
        dali_b_mean = float(dali_out[0, 2].mean())
        cpu_r_mean = float(cpu_out[0, 0].mean())
        cpu_b_mean = float(cpu_out[0, 2].mean())

        assert dali_r_mean == pytest.approx(cpu_r_mean, abs=0.02)
        assert dali_b_mean == pytest.approx(cpu_b_mean, abs=0.02)
        assert dali_r_mean < dali_b_mean

    def test_square_image_no_padding(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)
        dali_out, cpu_out = self._preprocess_both(image, (640, 640))
        assert dali_out.shape == (1, 3, 640, 640)
        np.testing.assert_allclose(dali_out, cpu_out, atol=0.02, rtol=0)
