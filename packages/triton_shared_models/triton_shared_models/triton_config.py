"""Triton model configuration and config.pbtxt rendering."""

from __future__ import annotations

from pydantic import BaseModel

from triton_shared_models.triton_types import (
    TritonDataType,
    TritonInstanceKind,
    TritonTensorSpec,
)


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


class TritonModelConfig(BaseModel):
    """Full Triton model configuration that renders to config.pbtxt format."""

    name: str
    platform: str = "onnxruntime_onnx"
    max_batch_size: int = 0
    inputs: list[TritonTensorSpec]
    outputs: list[TritonTensorSpec]
    instance_count: int = 1
    instance_kind: TritonInstanceKind = TritonInstanceKind.KIND_GPU
    default_model_filename: str = ""
    dynamic_batching: DynamicBatchingConfig | None = None
    optimization: OptimizationConfig | None = None
    model_warmup: list[ModelWarmupConfig] | None = None

    def to_pbtxt(self) -> str:
        """Render this configuration as Triton config.pbtxt protobuf text."""
        lines: list[str] = [
            f'name: "{self.name}"',
            f'platform: "{self.platform}"',
            f"max_batch_size: {self.max_batch_size}",
        ]

        if self.default_model_filename:
            lines.append(f'default_model_filename: "{self.default_model_filename}"')

        lines.append("")

        for tensor in self.inputs:
            lines.extend(_render_tensor("input", tensor))
            lines.append("")

        for tensor in self.outputs:
            lines.extend(_render_tensor("output", tensor))
            lines.append("")

        lines.extend(
            [
                "instance_group [",
                "  {",
                f"    count: {self.instance_count}",
                f"    kind: {self.instance_kind.value}",
                "  }",
                "]",
                "",
            ]
        )

        if self.dynamic_batching is not None:
            lines.extend(_render_dynamic_batching(self.dynamic_batching))
            lines.append("")

        if self.optimization is not None:
            lines.extend(_render_optimization(self.optimization))
            lines.append("")

        if self.model_warmup:
            for warmup in self.model_warmup:
                lines.extend(_render_model_warmup(warmup))
                lines.append("")

        return "\n".join(lines)


def _render_tensor(section: str, tensor: TritonTensorSpec) -> list[str]:
    dims_str = ", ".join(str(d) for d in tensor.dims)
    result = [
        f"{section} [",
        "  {",
        f'    name: "{tensor.name}"',
        f"    data_type: {tensor.data_type.value}",
        f"    dims: [ {dims_str} ]",
    ]
    if tensor.is_shape_tensor:
        result.append("    is_shape_tensor: true")
    result.extend(["  }", "]"])
    return result


def _render_dynamic_batching(config: DynamicBatchingConfig) -> list[str]:
    lines = ["dynamic_batching {"]
    if config.max_queue_delay_microseconds > 0:
        lines.append(f"  max_queue_delay_microseconds: {config.max_queue_delay_microseconds}")
    if config.preferred_batch_size:
        sizes = ", ".join(str(s) for s in config.preferred_batch_size)
        lines.append(f"  preferred_batch_size: [ {sizes} ]")
    lines.append("}")
    return lines


def _render_optimization(config: OptimizationConfig) -> list[str]:
    lines = ["optimization {"]
    if config.graph_level >= 0:
        lines.append(f"  graph_level: {config.graph_level}")
    if config.cuda_graphs:
        lines.append("  cuda { graphs: true }")
    lines.append("}")
    return lines


def _render_model_warmup(warmup: ModelWarmupConfig) -> list[str]:
    lines = [
        "model_warmup [",
        "  {",
        f'    name: "{warmup.name}"',
        f"    batch_size: {warmup.batch_size}",
        "    inputs {",
    ]
    for input_name, spec in warmup.inputs.items():
        dims_str = ", ".join(str(d) for d in spec.dims)
        lines.extend(
            [
                f'      key: "{input_name}"',
                "      value {",
                f"        data_type: {spec.data_type.value}",
                f"        dims: [ {dims_str} ]",
                f"        {'zero_data' if spec.zero_data else 'random_data'} {{}}",
                "      }",
            ]
        )
    lines.extend(
        [
            "    }",
            "  }",
            "]",
        ]
    )
    return lines
