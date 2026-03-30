"""Pydantic models for YOLO model registry and re-exports of shared Triton types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from triton_shared_models.triton_config import TritonModelConfig
from triton_shared_models.triton_types import (
    TritonDataType,
    TritonInstanceKind,
    TritonTensorSpec,
)

__all__ = [
    "YOLO_MODELS",
    "TritonDataType",
    "TritonInstanceKind",
    "TritonModelConfig",
    "TritonTensorSpec",
    "YoloModelVariant",
]


class YoloModelVariant(BaseModel):
    """Describes a single YOLO model variant to export for Triton."""

    name: str
    ultralytics_model: str
    export_kwargs: dict[str, Any] = {}
    imgsz: int = 640


YOLO_MODELS: tuple[YoloModelVariant, ...] = (
    YoloModelVariant(
        name="yolo26s",
        ultralytics_model="yolo26s.pt",
        export_kwargs={"end2end": True},
    ),
    YoloModelVariant(
        name="yolo11s",
        ultralytics_model="yolo11s.pt",
    ),
    YoloModelVariant(
        name="yolo12s",
        ultralytics_model="yolo12s.pt",
    ),
)
