"""Shared Triton type definitions for the triton-redis-connector workspace."""

from __future__ import annotations

from triton_shared_models.triton_config import TritonModelConfig
from triton_shared_models.triton_types import (
    TritonDataType,
    TritonInstanceKind,
    TritonTensorSpec,
)

__all__ = [
    "TritonDataType",
    "TritonInstanceKind",
    "TritonModelConfig",
    "TritonTensorSpec",
]
