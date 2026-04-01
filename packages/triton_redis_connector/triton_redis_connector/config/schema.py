"""Pydantic configuration models for the triton-redis-connector pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RedisConnectionConfig(BaseModel):
    """Redis server connection parameters."""

    model_config = ConfigDict(frozen=True)

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0)
    decode_responses: bool = False


class CollectorMode(StrEnum):
    """Redis Stream collector startup behavior."""

    RECOVERY = "recovery"
    FROM_LATEST = "from_latest"
    FROM_OLDEST = "from_oldest"


class SourceConfig(BaseModel):
    """Configuration for a single Redis Stream source."""

    model_config = ConfigDict(frozen=True)

    stream_name: str = Field(min_length=1)
    consumer_group: str = Field(min_length=1)
    consumer_name: str = ""
    event_model: str = Field(min_length=1)
    batch_size: int = Field(default=10, ge=1)
    block_ms: int = Field(default=1000, ge=0)
    newest_first_on_recovery: bool = True
    recovery_interleave_ratio: int = Field(default=4, ge=1)
    pending_ttl_s: float = Field(default=60.0, gt=0.0)
    collector_mode: CollectorMode = CollectorMode.RECOVERY


class SinkConfig(BaseModel):
    """Configuration for a single Redis Stream sink."""

    model_config = ConfigDict(frozen=True)

    stream_name: str = Field(min_length=1)
    maxlen: int | None = Field(default=10_000, ge=1)
    approximate_maxlen: bool = True


class DecoderType(StrEnum):
    """Supported image decoder backends."""

    OPENCV = "opencv"
    TURBOJPEG = "turbojpeg"


class DecoderPoolType(StrEnum):
    """Execution pool type for image decoding."""

    THREAD = "thread"
    PROCESS = "process"


class InferenceConfig(BaseModel):
    """Triton Inference Server connection and retry parameters."""

    model_config = ConfigDict(frozen=True)

    grpc_url: str = "localhost:8001"
    num_inference_workers: int = Field(default=2, ge=1)
    max_concurrent_per_worker: int = Field(default=8, ge=1)
    max_retry_attempts: int = Field(default=3, ge=0)
    retry_base_delay_s: float = Field(default=0.1, gt=0.0)
    retry_backoff_multiplier: float = Field(default=4.0, ge=1.0)
    model_version_poll_interval_s: float = Field(default=5.0, gt=0.0)


class PreprocessingConfig(BaseModel):
    """Image preprocessing pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    decoder: DecoderType = DecoderType.OPENCV
    decoder_pool_type: DecoderPoolType = DecoderPoolType.THREAD
    decoder_pool_workers: int = Field(default=4, ge=1)
    default_target_size: tuple[int, int] = (640, 640)


class QueueConfig(BaseModel):
    """Internal queue sizing configuration (applied to all queues)."""

    model_config = ConfigDict(frozen=True)

    maxsize: int = Field(default=1000, ge=1)


class MetricsConfig(BaseModel):
    """Metrics collection and emission configuration."""

    model_config = ConfigDict(frozen=True)

    output_stream: str = "metrics"
    emit_interval_s: float = Field(default=5.0, gt=0.0)
    triton_poll_interval_s: float = Field(default=10.0, gt=0.0)
    error_flush_interval_s: float = Field(default=30.0, gt=0.0)
    stats_log_interval_s: float = Field(default=60.0, gt=0.0)
    transit_time_enabled: bool = True


class DebugConfig(BaseModel):
    """Debug and development settings."""

    model_config = ConfigDict(frozen=True)

    stop_on_error: bool = False
    single_thread: bool = False


class RouterConfig(BaseModel):
    """Configuration for event routing to inference models."""

    model_config = ConfigDict(frozen=True)

    router_class: str = "triton_redis_connector.routers.default.FieldBasedRouter"
    default_model: str = ""
    model_field: str = "model_name"


class OutputRoutingConfig(BaseModel):
    """Maps model names to output Redis streams."""

    model_config = ConfigDict(frozen=True)

    default_stream: str | None = "detection_results"
    model_streams: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_at_least_one_routing_target(self) -> Self:
        """Ensure at least default_stream or model_streams is configured."""
        if not self.default_stream and not self.model_streams:
            msg = "OutputRoutingConfig requires at least one of 'default_stream' or 'model_streams' to be set"
            raise ValueError(msg)
        return self


class AppConfig(BaseModel):
    """Top-level application configuration, composed from Hydra YAML files."""

    model_config = ConfigDict(frozen=True)

    app_name: str = "triton-connector"
    redis: RedisConnectionConfig = RedisConnectionConfig()
    sources: list[SourceConfig] = []
    sinks: list[SinkConfig] = []
    processors: list[str] = []
    inference: InferenceConfig = InferenceConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    queues: QueueConfig = QueueConfig()
    metrics: MetricsConfig = MetricsConfig()
    debug: DebugConfig = DebugConfig()
    router: RouterConfig = RouterConfig()
    output_routing: OutputRoutingConfig = OutputRoutingConfig()
