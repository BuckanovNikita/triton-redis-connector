"""Triton model repository directory builder."""

import shutil
from pathlib import Path

from loguru import logger

from triton_model_tools.models.registry import TritonModelConfig

EXTENSION_TO_FILENAME: dict[str, str] = {
    ".onnx": "model.onnx",
    ".plan": "model.plan",
}


def build_model_directory(
    repository_root: Path,
    config: TritonModelConfig,
    model_source: Path,
    model_filename: str = "",
) -> Path:
    """Create Triton model directory with config.pbtxt and versioned model file.

    Creates the standard Triton repository structure:
        {repository_root}/{model_name}/config.pbtxt
        {repository_root}/{model_name}/1/{model_filename}

    Args:
        repository_root: Root path of the Triton model repository.
        config: Triton model configuration to write as config.pbtxt.
        model_source: Path to the exported model file (.onnx or .plan).
        model_filename: Target filename in version directory.
            If empty, derived from the source file extension.

    Returns:
        Path to the created model directory.

    """
    model_dir = repository_root / config.name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "config.pbtxt"
    config_path.write_text(config.to_pbtxt())
    logger.info("Wrote {}", config_path)

    resolved_filename = model_filename or EXTENSION_TO_FILENAME.get(model_source.suffix, model_source.name)
    target_model = version_dir / resolved_filename
    shutil.copy2(model_source, target_model)
    model_size_mb = target_model.stat().st_size / (1024 * 1024)
    logger.info("Copied model to {} ({:.1f} MB)", target_model, model_size_mb)

    return model_dir
