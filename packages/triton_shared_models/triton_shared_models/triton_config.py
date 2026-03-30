"""Triton model configuration and config.pbtxt rendering."""

from __future__ import annotations

from pydantic import BaseModel

from triton_shared_models.triton_types import (
    TritonInstanceKind,
    TritonTensorSpec,
)


class EnsembleStepMapping(BaseModel):
    """Maps a key from one ensemble step to another."""

    key: str
    value: str


class EnsembleStep(BaseModel):
    """A single step in a Triton ensemble pipeline."""

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
    ensemble_scheduling: list[EnsembleStep] | None = None

    def to_pbtxt(self) -> str:
        """Render this configuration as Triton config.pbtxt protobuf text."""
        lines: list[str] = [
            f'name: "{self.name}"',
        ]

        if self.backend is not None:
            lines.append(f'backend: "{self.backend}"')
        else:
            lines.append(f'platform: "{self.platform}"')

        lines.extend(
            [
                f"max_batch_size: {self.max_batch_size}",
                "",
            ]
        )

        for tensor in self.inputs:
            lines.extend(_render_tensor("input", tensor))
            lines.append("")

        for tensor in self.outputs:
            lines.extend(_render_tensor("output", tensor))
            lines.append("")

        if self.ensemble_scheduling is not None:
            lines.extend(_render_ensemble_scheduling(self.ensemble_scheduling))
            lines.append("")
        else:
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

        return "\n".join(lines)


def _render_tensor(section: str, tensor: TritonTensorSpec) -> list[str]:
    dims_str = ", ".join(str(d) for d in tensor.dims)
    return [
        f"{section} [",
        "  {",
        f'    name: "{tensor.name}"',
        f"    data_type: {tensor.data_type.value}",
        f"    dims: [ {dims_str} ]",
        "  }",
        "]",
    ]


def _render_ensemble_scheduling(steps: list[EnsembleStep]) -> list[str]:
    lines: list[str] = ["ensemble_scheduling {"]
    for step in steps:
        lines.append("  step [")
        lines.append("    {")
        lines.append(f'      model_name: "{step.model_name}"')
        lines.append(f"      model_version: {step.model_version}")
        for mapping in step.input_map:
            lines.append("      input_map {")
            lines.append(f'        key: "{mapping.key}"')
            lines.append(f'        value: "{mapping.value}"')
            lines.append("      }")
        for mapping in step.output_map:
            lines.append("      output_map {")
            lines.append(f'        key: "{mapping.key}"')
            lines.append(f'        value: "{mapping.value}"')
            lines.append("      }")
        lines.append("    }")
        lines.append("  ]")
    lines.append("}")
    return lines
