"""Configuration models and Hydra integration for the connector pipeline."""

from __future__ import annotations

from triton_redis_connector.config.hydra_app import load_config
from triton_redis_connector.config.schema import (
    AppConfig,
    CollectorMode,
    DecoderPoolType,
    DecoderType,
    InferenceConfig,
    MetricsConfig,
    PreprocessingConfig,
    QueueConfig,
    RedisConnectionConfig,
    SinkConfig,
    SourceConfig,
)

__all__ = [
    "AppConfig",
    "CollectorMode",
    "DecoderPoolType",
    "DecoderType",
    "InferenceConfig",
    "MetricsConfig",
    "PreprocessingConfig",
    "QueueConfig",
    "RedisConnectionConfig",
    "SinkConfig",
    "SourceConfig",
    "load_config",
]
