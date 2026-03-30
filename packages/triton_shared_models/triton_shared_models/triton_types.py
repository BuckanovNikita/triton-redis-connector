"""Triton Inference Server data types and tensor specifications."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class TritonDataType(StrEnum):
    """Triton Inference Server data types."""

    TYPE_FP32 = "TYPE_FP32"
    TYPE_FP16 = "TYPE_FP16"
    TYPE_INT8 = "TYPE_INT8"
    TYPE_INT16 = "TYPE_INT16"
    TYPE_INT32 = "TYPE_INT32"
    TYPE_INT64 = "TYPE_INT64"
    TYPE_UINT8 = "TYPE_UINT8"
    TYPE_BOOL = "TYPE_BOOL"


class TritonInstanceKind(StrEnum):
    """Triton instance group kind."""

    KIND_GPU = "KIND_GPU"
    KIND_CPU = "KIND_CPU"


class TritonTensorSpec(BaseModel):
    """Specification for a single Triton input or output tensor."""

    name: str
    data_type: TritonDataType
    dims: list[int]
