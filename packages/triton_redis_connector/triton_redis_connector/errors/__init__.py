"""Error event models for the triton-redis-connector pipeline."""

from triton_redis_connector.errors.models import (
    InferenceError,
    PreprocessingError,
)

__all__ = [
    "InferenceError",
    "PreprocessingError",
]
