"""Inference event models for the Triton inference pipeline."""

from __future__ import annotations

from triton_redis_connector.inference.request import InferenceInput, InferenceOutput

__all__ = [
    "InferenceInput",
    "InferenceOutput",
]
