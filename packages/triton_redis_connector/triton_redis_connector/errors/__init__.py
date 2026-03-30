"""Error event models and handling for the connector pipeline."""

from __future__ import annotations

from triton_redis_connector.errors.collector import ErrorCollector
from triton_redis_connector.errors.handler import ErrorHandler
from triton_redis_connector.errors.models import (
    ConnectorError,
    InferenceError,
    PreprocessingError,
)

__all__ = [
    "ConnectorError",
    "ErrorCollector",
    "ErrorHandler",
    "InferenceError",
    "PreprocessingError",
]
