"""Pydantic configuration models for the triton-redis-connector pipeline."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Any, Self

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
    recovery_enabled: bool = True
    recovery_max_rps: int = Field(default=10, ge=1)


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


class NodeVariant(StrEnum):
    """Execution variant for config-driven pipeline nodes (parallel only).

    Values match ``ExecutionVariant`` in ``engine.registry`` so the assembly
    layer can convert between them with a simple ``ExecutionVariant(value)``.
    """

    THREAD_SOURCE = auto()
    PROCESS_SOURCE = auto()
    THREAD_PROCESSOR = auto()
    THREAD_POOL_PROCESSOR = auto()
    PROCESS_POOL_PROCESSOR = auto()
    THREAD_SINK = auto()


_POOL_VARIANTS = {NodeVariant.THREAD_POOL_PROCESSOR, NodeVariant.PROCESS_POOL_PROCESSOR}


class NodeConfig(BaseModel):
    """Configuration for a single pipeline node assembled from config.

    Used only for user-defined processors that need ``class_path``.
    Shipped processors are registered via ``Application.build()``.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    class_path: str = Field(min_length=1)
    variant: NodeVariant
    params: dict[str, Any] = Field(default_factory=dict)
    tracked: bool = False
    tracked_name: str = ""
    pool_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def pool_size_requires_pool_variant(self) -> Self:
        """Reject pool_size > 1 for non-pool variants."""
        if self.pool_size > 1 and self.variant not in _POOL_VARIANTS:
            msg = f"pool_size > 1 requires a pool variant, got {self.variant!r}"
            raise ValueError(msg)
        return self


class PostprocessorConfig(BaseModel):
    """YOLO postprocessor thresholds for shipped YoloPostProcessorBridge."""

    model_config = ConfigDict(frozen=True)

    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    end2end: bool = False


class PipelineConfig(BaseModel):
    """Shipped processor configuration resolved by Application.build().

    Infrastructure params for shipped processors. All class references
    (event model, user processors) come from code, not config.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str = ""
    model_input_name: str = "images"
    model_output_names: list[str] = Field(default_factory=lambda: ["output0"])
    postprocessor: PostprocessorConfig = PostprocessorConfig()


class AppConfig(BaseModel):
    """Top-level application configuration, composed from Hydra YAML files."""

    model_config = ConfigDict(frozen=True)

    app_name: str = "triton-connector"
    redis: RedisConnectionConfig = RedisConnectionConfig()
    sources: list[SourceConfig] = []
    sinks: list[SinkConfig] = []
    nodes: list[NodeConfig] = []
    pipeline: PipelineConfig = PipelineConfig()
    inference: InferenceConfig = InferenceConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    queues: QueueConfig = QueueConfig()
    metrics: MetricsConfig = MetricsConfig()
    debug: DebugConfig = DebugConfig()
