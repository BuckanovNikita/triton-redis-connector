"""NMS postprocessing model generation for Triton Python backend."""

import shutil
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from triton_model_tools.models import (
    TritonDataType,
    TritonInstanceKind,
    TritonModelConfig,
    TritonTensorSpec,
)

NMS_MODEL_NAME = "nms_postprocess"
_NMS_SCRIPT_PATH = Path(__file__).parent / "scripts" / "nms_postprocess.py"


class NmsModelConfig(BaseModel):
    """Configuration for NMS postprocessing model generation."""

    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=300, ge=1)


def build_nms_model_config(config: NmsModelConfig) -> TritonModelConfig:
    """Build a TritonModelConfig for the NMS postprocessing model.

    Args:
        config: NMS configuration parameters.

    Returns:
        TritonModelConfig with Python backend settings for NMS.

    """
    return TritonModelConfig(
        name=NMS_MODEL_NAME,
        backend="python",
        max_batch_size=0,
        inputs=[
            TritonTensorSpec(
                name="INPUT_0",
                data_type=TritonDataType.TYPE_FP32,
                dims=[-1, -1],
            ),
        ],
        outputs=[
            TritonTensorSpec(
                name="OUTPUT_0",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1, -1, 6],
            ),
        ],
        instance_kind=TritonInstanceKind.KIND_CPU,
        parameters={
            "confidence_threshold": str(config.confidence_threshold),
            "iou_threshold": str(config.iou_threshold),
            "max_detections": str(config.max_detections),
        },
    )


def build_nms_model_directory(
    repository_root: Path,
    config: NmsModelConfig,
) -> Path:
    """Create the Triton model repository directory for NMS postprocessing.

    Creates the directory structure:
        nms_postprocess/1/model.py
        nms_postprocess/config.pbtxt

    Args:
        repository_root: Root of the Triton model repository.
        config: NMS configuration parameters.

    Returns:
        Path to the created NMS model directory.

    """
    model_dir = repository_root / NMS_MODEL_NAME
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "model.py"
    shutil.copy2(_NMS_SCRIPT_PATH, model_path)
    logger.info("Copied NMS model script to {}", model_path)

    model_config = build_nms_model_config(config)
    config_path = model_dir / "config.pbtxt"
    config_path.write_text(model_config.to_pbtxt())
    logger.info("Wrote NMS config to {}", config_path)

    return model_dir
