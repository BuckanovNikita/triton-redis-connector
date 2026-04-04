"""Pydantic configuration models for the triton-redis-connector pipeline."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from typed_flow.config.schema import AppConfig


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
    metrics_poll_interval_s: float = Field(default=10.0, gt=0.0)


class PreprocessingConfig(BaseModel):
    """Image preprocessing pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    decoder: DecoderType = DecoderType.OPENCV
    decoder_pool_type: DecoderPoolType = DecoderPoolType.PROCESS
    decoder_pool_workers: int = Field(default=8, ge=1)
    default_target_size: tuple[int, int] = (640, 640)

    @property
    def num_preprocess_workers(self) -> int:
        """Resolve num_workers: THREAD=1 (no pool overhead), PROCESS=pool_workers."""
        if self.decoder_pool_type == DecoderPoolType.THREAD:
            return 1
        return self.decoder_pool_workers


class PipelineConfig(BaseModel):
    """Shipped processor configuration resolved by Application.build().

    Infrastructure params for shipped processors. All class references
    (event model, user processors) come from code, not config.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str = ""
    model_input_name: str = "images"
    model_output_names: list[str] = Field(default_factory=lambda: ["output0"])


class TritonAppConfig(AppConfig):
    """Triton-specific application configuration extending the base typed-flow config."""

    model_config = ConfigDict(frozen=True)

    inference: InferenceConfig = InferenceConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    pipeline: PipelineConfig = PipelineConfig()
