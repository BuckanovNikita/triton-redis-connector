"""Triton model configuration and config.pbtxt rendering."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
from triton_shared_models.triton_types import (
    TritonDataType,
    TritonInstanceKind,
    TritonTensorSpec,
)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    lstrip_blocks=True,
    trim_blocks=True,
    keep_trailing_newline=False,
    autoescape=False,
)
_CONFIG_TEMPLATE = _JINJA_ENV.get_template("config.pbtxt.j2")


class WarmupInputSpec(BaseModel):
    """Specification for a single warmup input tensor."""

    data_type: TritonDataType
    dims: list[int]
    zero_data: bool = True


class ModelWarmupConfig(BaseModel):
    """Configuration for a single model warmup entry."""

    name: str
    batch_size: int
    inputs: dict[str, WarmupInputSpec]


class DynamicBatchingConfig(BaseModel):
    """Configuration for Triton dynamic batching."""

    max_queue_delay_microseconds: int = 0
    preferred_batch_size: list[int] = []


class OptimizationConfig(BaseModel):
    """Configuration for Triton model optimization."""

    graph_level: int = -1
    cuda_graphs: bool = False
    trt_acceleration: bool = False


class EnsembleStepMapping(BaseModel):
    """Maps an ensemble model I/O name to a step's I/O name."""

    key: str
    value: str


class EnsembleStep(BaseModel):
    """A single step in a Triton ensemble schedule."""

    model_name: str
    model_version: int = -1
    input_map: list[EnsembleStepMapping] = []
    output_map: list[EnsembleStepMapping] = []


class TritonModelConfig(BaseModel):
    """Full Triton model configuration that renders to config.pbtxt format."""

    name: str
    platform: str = "onnxruntime_onnx"
    backend: str | None = None
    max_batch_size: int = 0
    inputs: list[TritonTensorSpec]
    outputs: list[TritonTensorSpec]
    instance_count: int = 1
    instance_kind: TritonInstanceKind = TritonInstanceKind.KIND_GPU
    default_model_filename: str = ""
    dynamic_batching: DynamicBatchingConfig | None = None
    optimization: OptimizationConfig | None = None
    model_warmup: list[ModelWarmupConfig] | None = None
    ensemble_scheduling: list[EnsembleStep] | None = None
    parameters: dict[str, str] = {}

    def to_pbtxt(self) -> str:
        """Render this configuration as Triton config.pbtxt protobuf text."""
        return _CONFIG_TEMPLATE.render(config=self).strip()
