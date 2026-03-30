"""Triton model repository directory builder."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

    from triton_model_tools.models import TritonModelConfig


def build_model_directory(
    repository_root: Path,
    config: TritonModelConfig,
    onnx_source: Path,
) -> Path:
    """Create Triton model directory with config.pbtxt and versioned ONNX file.

    Creates the standard Triton repository structure:
        {repository_root}/{model_name}/config.pbtxt
        {repository_root}/{model_name}/1/model.onnx

    Args:
        repository_root: Root path of the Triton model repository.
        config: Triton model configuration to write as config.pbtxt.
        onnx_source: Path to the exported ONNX model file.

    Returns:
        Path to the created model directory.

    """
    model_dir = repository_root / config.name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "config.pbtxt"
    config_path.write_text(config.to_pbtxt())
    logger.info("Wrote {}", config_path)

    target_onnx = version_dir / "model.onnx"
    shutil.copy2(onnx_source, target_onnx)
    onnx_size_mb = target_onnx.stat().st_size / (1024 * 1024)
    logger.info("Copied ONNX model to {} ({:.1f} MB)", target_onnx, onnx_size_mb)

    return model_dir
