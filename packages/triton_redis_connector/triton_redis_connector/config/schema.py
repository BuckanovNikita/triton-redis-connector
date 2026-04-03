"""Pydantic configuration models for the triton-redis-connector pipeline."""

from __future__ import annotations

from enum import StrEnum

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
    block_ms: int = Field(default=1000, ge=0)


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
    DALI = "dali"


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
    mermaid_path: str = ""


class PipelineConfig(BaseModel):
    """Shipped processor configuration resolved by Application.build().

    Infrastructure params for shipped processors. All class references
    (event model, user processors) come from code, not config.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str = ""
    model_input_name: str = "images"
    model_output_names: list[str] = Field(default_factory=lambda: ["output0"])


class AppConfig(BaseModel):
    """Top-level application configuration, composed from Hydra YAML files."""

    model_config = ConfigDict(frozen=True)

    redis: RedisConnectionConfig = RedisConnectionConfig()
    sources: list[SourceConfig] = []
    sinks: list[SinkConfig] = []
    pipeline: PipelineConfig = PipelineConfig()
    inference: InferenceConfig = InferenceConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    queues: QueueConfig = QueueConfig()
    metrics: MetricsConfig = MetricsConfig()
    debug: DebugConfig = DebugConfig()
