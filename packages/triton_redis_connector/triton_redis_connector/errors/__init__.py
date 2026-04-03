"""Error event models for the triton-redis-connector pipeline."""

from __future__ import annotations

from triton_redis_connector.errors.models import (
    InferenceError,
    PreprocessingError,
)

__all__ = [
    "InferenceError",
    "PreprocessingError",
]
