"""Tests for resolution validation and end2end NMS behavior in convert pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError
from triton_model_tools.export.dali import (
    DALI_ENSEMBLE_SUFFIX,
    DALI_NMS_ENSEMBLE_SUFFIX,
    DaliPipelineConfig,
)
from triton_model_tools.models import YoloModelVariant

if TYPE_CHECKING:
    from pathlib import Path


class TestResolutionValidationAcceptsValid:
    @pytest.mark.parametrize(
        ("imgsz", "expected_resolved"),
        [
            (640, (640, 640)),
            (512, (512, 512)),
            ((960, 608), (960, 608)),
            ((512, 512), (512, 512)),
        ],
        ids=[
            "standard-square-640",
            "non-standard-square-512",
            "rectangular-960x608",
            "explicit-tuple-512x512",
        ],
    )
    def test_valid_resolutions_accepted(
        self,
        imgsz: int | tuple[int, int],
        expected_resolved: tuple[int, int],
    ) -> None:
        variant = YoloModelVariant(
            name="test_model",
            ultralytics_model="test.pt",
            imgsz=imgsz,
        )
        assert variant.resolved_imgsz == expected_resolved


class TestResolutionValidationRejectsInvalid:
    @pytest.mark.parametrize(
        "imgsz",
        [
            (601, 612),
            (639, 640),
        ],
        ids=[
            "both-not-div32-601x612",
            "first-not-div32-639x640",
        ],
    )
    def test_invalid_resolutions_rejected(self, imgsz: tuple[int, int]) -> None:
        with pytest.raises(ValidationError, match="not divisible by 32"):
            YoloModelVariant(
                name="test_model",
                ultralytics_model="test.pt",
                imgsz=imgsz,
            )


class TestDaliConfigUsesCustomSize:
    def test_dali_config_uses_512_not_default_640(self) -> None:
        imgsz = 512
        resolved = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        dali_config = DaliPipelineConfig(target_size=resolved)

        assert dali_config.target_size == (512, 512)


class TestEnd2endNmsExclusion:
    def test_end2end_model_excludes_nms_from_ensemble(self, tmp_path: Path) -> None:
        from triton_model_tools.cli.convert import _generate_postprocess_models

        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        model_dir = repo_root / "yolo11s"
        model_dir.mkdir(parents=True)
        exported_dirs = [model_dir]

        _generate_postprocess_models(
            repo_root,
            exported_dirs,
            generate_nms=True,
            generate_rescale=False,
            generate_dali=True,
            end2end=True,
            target_size=(640, 640),
        )

        ensemble_dir = repo_root / f"yolo11s{DALI_ENSEMBLE_SUFFIX}"
        assert ensemble_dir.is_dir(), f"Expected 2-step ensemble dir {ensemble_dir} (without NMS)"

        nms_ensemble_dir = repo_root / f"yolo11s{DALI_NMS_ENSEMBLE_SUFFIX}"
        assert not nms_ensemble_dir.exists(), f"NMS ensemble dir {nms_ensemble_dir} should NOT exist for end2end model"

    def test_non_end2end_model_includes_nms_in_ensemble(self, tmp_path: Path) -> None:
        from triton_model_tools.cli.convert import _generate_postprocess_models

        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        model_dir = repo_root / "yolo11s"
        model_dir.mkdir(parents=True)
        exported_dirs = [model_dir]

        _generate_postprocess_models(
            repo_root,
            exported_dirs,
            generate_nms=True,
            generate_rescale=False,
            generate_dali=True,
            end2end=False,
            target_size=(640, 640),
        )

        nms_ensemble_dir = repo_root / f"yolo11s{DALI_NMS_ENSEMBLE_SUFFIX}"
        assert nms_ensemble_dir.is_dir(), f"Expected 3-step NMS ensemble dir {nms_ensemble_dir} for non-end2end model"

        plain_ensemble_dir = repo_root / f"yolo11s{DALI_ENSEMBLE_SUFFIX}"
        assert not plain_ensemble_dir.exists(), "Plain 2-step ensemble should NOT exist when NMS is enabled"

    def test_end2end_ensemble_config_has_no_nms_step(self, tmp_path: Path) -> None:
        from triton_model_tools.cli.convert import _generate_postprocess_models

        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        model_dir = repo_root / "yolo11s"
        model_dir.mkdir(parents=True)
        exported_dirs = [model_dir]

        _generate_postprocess_models(
            repo_root,
            exported_dirs,
            generate_nms=True,
            generate_rescale=False,
            generate_dali=True,
            end2end=True,
            target_size=(640, 640),
        )

        ensemble_config_path = repo_root / f"yolo11s{DALI_ENSEMBLE_SUFFIX}" / "config.pbtxt"
        config_text = ensemble_config_path.read_text()
        assert "nms_postprocess" not in config_text
