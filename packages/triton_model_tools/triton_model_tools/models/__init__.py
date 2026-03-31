"""Shared types: YOLO model registry and detection result models."""

from triton_model_tools.models.detection import (
    Detection,
    FolderInferenceResult,
    ImageDetectionResult,
)
from triton_model_tools.models.registry import (
    EXPORT_FORMAT_TO_FILENAME,
    EXPORT_FORMAT_TO_PLATFORM,
    YOLO_MODELS,
    DynamicBatchingConfig,
    EnsembleStep,
    EnsembleStepMapping,
    ExportFormat,
    ModelWarmupConfig,
    OptimizationConfig,
    TritonDataType,
    TritonInstanceKind,
    TritonModelConfig,
    TritonTensorSpec,
    WarmupInputSpec,
    YoloModelVariant,
)

__all__ = [
    "EXPORT_FORMAT_TO_FILENAME",
    "EXPORT_FORMAT_TO_PLATFORM",
    "YOLO_MODELS",
    "Detection",
    "DynamicBatchingConfig",
    "EnsembleStep",
    "EnsembleStepMapping",
    "ExportFormat",
    "FolderInferenceResult",
    "ImageDetectionResult",
    "ModelWarmupConfig",
    "OptimizationConfig",
    "TritonDataType",
    "TritonInstanceKind",
    "TritonModelConfig",
    "TritonTensorSpec",
    "WarmupInputSpec",
    "YoloModelVariant",
]
