"""Rescale postprocessing model generation for Triton Python backend."""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger

from triton_model_tools.models import (
    TritonDataType,
    TritonInstanceKind,
    TritonModelConfig,
    TritonTensorSpec,
)

RESCALE_MODEL_NAME = "rescale_postprocess"
_RESCALE_SCRIPT_PATH = Path(__file__).parent / "scripts" / "rescale_postprocess.py"


def build_rescale_model_config() -> TritonModelConfig:
    """Build a TritonModelConfig for the rescale postprocessing model.

    The rescale model transforms detection coordinates from letterbox space
    back to original image coordinates using: (coord - pad) / scale.

    Returns:
        TritonModelConfig with Python backend settings for rescale.

    """
    return TritonModelConfig(
        name=RESCALE_MODEL_NAME,
        backend="python",
        max_batch_size=0,
        inputs=[
            TritonTensorSpec(
                name="DETECTIONS",
                data_type=TritonDataType.TYPE_FP32,
                dims=[-1, -1],
            ),
            TritonTensorSpec(
                name="SCALE",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1],
            ),
            TritonTensorSpec(
                name="PAD_LEFT",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1],
            ),
            TritonTensorSpec(
                name="PAD_TOP",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1],
            ),
        ],
        outputs=[
            TritonTensorSpec(
                name="RESCALED_DETECTIONS",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1, -1, 6],
            ),
        ],
        instance_kind=TritonInstanceKind.KIND_CPU,
    )


def build_rescale_model_directory(repository_root: Path) -> Path:
    """Create the Triton model repository directory for rescale postprocessing.

    Creates the directory structure:
        rescale_postprocess/1/model.py
        rescale_postprocess/config.pbtxt

    Args:
        repository_root: Root of the Triton model repository.

    Returns:
        Path to the created rescale model directory.

    """
    model_dir = repository_root / RESCALE_MODEL_NAME
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "model.py"
    shutil.copy2(_RESCALE_SCRIPT_PATH, model_path)
    logger.info("Copied rescale model script to {}", model_path)

    model_config = build_rescale_model_config()
    config_path = model_dir / "config.pbtxt"
    config_path.write_text(model_config.to_pbtxt())
    logger.info("Wrote rescale config to {}", config_path)

    return model_dir
