"""Model repository validation: check directory structure and config.pbtxt integrity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

REQUIRED_CONFIG_FILENAME = "config.pbtxt"
VERSION_DIR_NAME = "1"
MODEL_EXTENSIONS = frozenset({".onnx", ".plan", ".savedmodel", ".pt", ".py"})


def validate_model_directory(model_dir: Path) -> list[str]:
    """Validate a single Triton model directory structure.

    Checks:
    - config.pbtxt exists and is non-empty
    - Version directory (1/) exists
    - At least one model file exists in version directory
    - config.pbtxt contains required fields (name, platform or backend)

    Args:
        model_dir: Path to the model directory.

    Returns:
        List of validation error strings. Empty list means valid.

    """
    errors: list[str] = []
    model_name = model_dir.name

    config_path = model_dir / REQUIRED_CONFIG_FILENAME
    if not config_path.exists():
        errors.append(f"{model_name}: missing {REQUIRED_CONFIG_FILENAME}")
        return errors

    config_text = config_path.read_text().strip()
    if not config_text:
        errors.append(f"{model_name}: {REQUIRED_CONFIG_FILENAME} is empty")
        return errors

    if "name:" not in config_text:
        errors.append(f"{model_name}: {REQUIRED_CONFIG_FILENAME} missing 'name' field")

    if "platform:" not in config_text and "backend:" not in config_text:
        errors.append(f"{model_name}: {REQUIRED_CONFIG_FILENAME} missing 'platform' or 'backend' field")

    if "input" not in config_text:
        errors.append(f"{model_name}: {REQUIRED_CONFIG_FILENAME} missing input specification")

    if "output" not in config_text:
        errors.append(f"{model_name}: {REQUIRED_CONFIG_FILENAME} missing output specification")

    version_dir = model_dir / VERSION_DIR_NAME
    if not version_dir.is_dir():
        errors.append(f"{model_name}: missing version directory '{VERSION_DIR_NAME}/'")
    else:
        model_files = [f for f in version_dir.iterdir() if f.is_file()]
        if not model_files:
            errors.append(f"{model_name}: no model files in '{VERSION_DIR_NAME}/'")

    return errors


def validate_repository(repository_root: Path) -> dict[str, list[str]]:
    """Validate all model directories in a Triton model repository.

    Args:
        repository_root: Root path of the Triton model repository.

    Returns:
        Dict mapping model name to list of validation errors.
        Models with no errors are included with empty lists.

    """
    if not repository_root.is_dir():
        logger.error("Repository root does not exist: {}", repository_root)
        return {"_repository": [f"Repository root does not exist: {repository_root}"]}

    results: dict[str, list[str]] = {}

    for model_dir in sorted(repository_root.iterdir()):
        if not model_dir.is_dir():
            continue

        errors = validate_model_directory(model_dir)
        results[model_dir.name] = errors

        if errors:
            for error in errors:
                logger.warning("Validation error: {}", error)
        else:
            logger.debug("Model {} passed validation", model_dir.name)

    valid_count = sum(1 for errs in results.values() if not errs)
    invalid_count = sum(1 for errs in results.values() if errs)
    logger.info(
        "Repository validation: {} valid, {} invalid out of {} models",
        valid_count,
        invalid_count,
        len(results),
    )

    return results
