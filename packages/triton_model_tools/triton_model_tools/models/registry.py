"""Pydantic models for YOLO model registry and re-exports of shared Triton types."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel
from triton_shared_models.triton_config import (
    DynamicBatchingConfig,
    EnsembleStep,
    EnsembleStepMapping,
    ModelWarmupConfig,
    OptimizationConfig,
    TritonModelConfig,
    WarmupInputSpec,
)
from triton_shared_models.triton_types import (
    EXPORT_FORMAT_TO_FILENAME,
    EXPORT_FORMAT_TO_PLATFORM,
    ExportFormat,
    TritonDataType,
    TritonInstanceKind,
    TritonTensorSpec,
)

__all__ = [
    "EXPORT_FORMAT_TO_FILENAME",
    "EXPORT_FORMAT_TO_PLATFORM",
    "YOLO_MODELS",
    "DiscoveredModel",
    "DynamicBatchingConfig",
    "EnsembleStep",
    "EnsembleStepMapping",
    "ExportFormat",
    "ModelWarmupConfig",
    "OptimizationConfig",
    "TritonDataType",
    "TritonInstanceKind",
    "TritonModelConfig",
    "TritonTensorSpec",
    "WarmupInputSpec",
    "YoloModelVariant",
    "discover_models",
    "load_export_config",
]

_SKIP_SUFFIXES = ("_dali_", "_ensemble", "_nms", "_trt", "_preprocess")
_PLATFORM_RE = re.compile(r'^platform:\s*"(.+?)"', re.MULTILINE)
_DIMS_RE = re.compile(r"dims:\s*\[\s*([\d\s,\-]+)\s*]")
_END2END_FEATURE_COUNT = 6


class YoloModelVariant(BaseModel):
    """Describes a single YOLO model variant to export for Triton."""

    name: str
    ultralytics_model: str
    export_kwargs: dict[str, Any] = {}
    imgsz: int = 640
    end2end_export: bool = False


class DiscoveredModel(BaseModel):
    """A model discovered by scanning a Triton model repository."""

    name: str
    platform: str
    input_dims: list[int]
    output_dims: list[int]
    is_end2end: bool


def _parse_dims_blocks(text: str, section: str) -> list[int]:
    pattern = re.compile(
        rf"{section}\s*\[\s*\{{(.*?)\}}",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return []
    block = match.group(1)
    dims_match = _DIMS_RE.search(block)
    if not dims_match:
        return []
    return [int(d.strip()) for d in dims_match.group(1).split(",") if d.strip()]


def discover_models(repository_root: Path) -> list[DiscoveredModel]:
    """Scan subdirectories of a Triton model repository and return discovered models.

    Skips directories containing DALI, ensemble, NMS, TRT, or preprocess suffixes.
    Parses config.pbtxt for platform and input/output dimensions.

    Args:
        repository_root: Root path of the Triton model repository.

    Returns:
        List of discovered model descriptors.

    """
    discovered: list[DiscoveredModel] = []

    if not repository_root.is_dir():
        logger.warning("Repository root does not exist: {}", repository_root)
        return discovered

    for model_dir in sorted(repository_root.iterdir()):
        if not model_dir.is_dir():
            continue

        dir_name = model_dir.name
        if any(suffix in dir_name for suffix in _SKIP_SUFFIXES):
            logger.debug("Skipping auxiliary model directory: {}", dir_name)
            continue

        config_path = model_dir / "config.pbtxt"
        if not config_path.exists():
            logger.debug("No config.pbtxt in {}, skipping", dir_name)
            continue

        config_text = config_path.read_text()

        platform_match = _PLATFORM_RE.search(config_text)
        if not platform_match:
            logger.warning("No platform found in {}/config.pbtxt, skipping", dir_name)
            continue

        platform = platform_match.group(1)
        input_dims = _parse_dims_blocks(config_text, "input")
        output_dims = _parse_dims_blocks(config_text, "output")
        is_end2end = len(output_dims) >= 3 and output_dims[-1] == _END2END_FEATURE_COUNT

        discovered.append(
            DiscoveredModel(
                name=dir_name,
                platform=platform,
                input_dims=input_dims,
                output_dims=output_dims,
                is_end2end=is_end2end,
            )
        )
        logger.debug(
            "Discovered model: {} (platform={}, end2end={})",
            dir_name,
            platform,
            is_end2end,
        )

    logger.info("Discovered {} model(s) in {}", len(discovered), repository_root)
    return discovered


def load_export_config(config_path: Path) -> list[YoloModelVariant]:
    """Load YOLO model export configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        List of YOLO model variants parsed from the config.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML structure is invalid.

    """
    if not config_path.exists():
        msg = f"Models config file not found: {config_path}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(config_path.read_text())

    if not isinstance(raw, dict) or "models" not in raw:
        msg = f"Invalid models config: expected top-level 'models' key in {config_path}"
        raise ValueError(msg)

    models_data: list[dict[str, Any]] = raw["models"]
    return [YoloModelVariant.model_validate(entry) for entry in models_data]


_DEFAULT_MODELS_PATH = Path(__file__).parent / "default_models.yaml"
YOLO_MODELS: tuple[YoloModelVariant, ...] = tuple(load_export_config(_DEFAULT_MODELS_PATH))
