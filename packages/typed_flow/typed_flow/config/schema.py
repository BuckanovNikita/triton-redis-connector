"""Pydantic configuration models for the typed-flow pipeline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RedisConnectionConfig(BaseModel):
    """Redis server connection parameters."""

    model_config = ConfigDict(frozen=True)

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0)
    decode_responses: bool = False


class SourceConfig(BaseModel):
    """Configuration for a single Redis Stream source."""

    model_config = ConfigDict(frozen=True)

    stream_name: str = Field(min_length=1)
    consumer_group: str = Field(min_length=1)
    consumer_name: str = ""
    event_model: str = Field(min_length=1)
    batch_size: int = Field(default=10, ge=1)
    block_ms: int = Field(default=0, ge=0)


class SinkConfig(BaseModel):
    """Configuration for a single Redis Stream sink."""

    model_config = ConfigDict(frozen=True)

    stream_name: str = Field(min_length=1)
    maxlen: int | None = Field(default=10_000, ge=1)
    approximate_maxlen: bool = True
    block_ms: int = Field(default=0, ge=0)


class QueueConfig(BaseModel):
    """Internal queue sizing configuration (applied to all queues)."""

    model_config = ConfigDict(frozen=True)

    maxsize: int = Field(default=50_000, ge=1)


class MetricsConfig(BaseModel):
    """Metrics collection and emission configuration."""

    model_config = ConfigDict(frozen=True)

    output_stream: str = "metrics"
    emit_interval_s: float = Field(default=5.0, gt=0.0)
    error_flush_interval_s: float = Field(default=30.0, gt=0.0)
    stats_log_interval_s: float = Field(default=60.0, gt=0.0)
    transit_time_enabled: bool = True


class DebugConfig(BaseModel):
    """Debug and development settings."""

    model_config = ConfigDict(frozen=True)

    stop_on_error: bool = False
    single_thread: bool = False
    mermaid_path: str = ""


class AppConfig(BaseModel):
    """Top-level application configuration for typed-flow pipelines."""

    model_config = ConfigDict(frozen=True)

    redis: RedisConnectionConfig = RedisConnectionConfig()
    sources: list[SourceConfig] = []
    sinks: list[SinkConfig] = []
    queues: QueueConfig = QueueConfig()
    metrics: MetricsConfig = MetricsConfig()
    debug: DebugConfig = DebugConfig()
