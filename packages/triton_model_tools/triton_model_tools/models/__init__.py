"""Shared types: YOLO model registry, detection and classification result models."""

from triton_model_tools.models.classification import (
    ClassificationPrediction,
    FolderClassificationResult,
    ImageClassificationResult,
)
from triton_model_tools.models.detection import (
    Detection,
    FolderInferenceResult,
    ImageDetectionResult,
)
from triton_model_tools.models.registry import (
    EXPORT_FORMAT_TO_FILENAME,
    EXPORT_FORMAT_TO_PLATFORM,
    YOLO_MODELS,
    DiscoveredModel,
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
    discover_models,
    load_export_config,
)

__all__ = [
    "EXPORT_FORMAT_TO_FILENAME",
    "EXPORT_FORMAT_TO_PLATFORM",
    "YOLO_MODELS",
    "ClassificationPrediction",
    "Detection",
    "DiscoveredModel",
    "DynamicBatchingConfig",
    "EnsembleStep",
    "EnsembleStepMapping",
    "ExportFormat",
    "FolderClassificationResult",
    "FolderInferenceResult",
    "ImageClassificationResult",
    "ImageDetectionResult",
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
