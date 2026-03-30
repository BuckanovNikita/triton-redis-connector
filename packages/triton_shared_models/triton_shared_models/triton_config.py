"""Triton model configuration and config.pbtxt rendering."""

from __future__ import annotations

from pydantic import BaseModel

from triton_shared_models.triton_types import (
    TritonInstanceKind,
    TritonTensorSpec,
)


class TritonModelConfig(BaseModel):
    """Full Triton model configuration that renders to config.pbtxt format."""

    name: str
    platform: str = "onnxruntime_onnx"
    max_batch_size: int = 0
    inputs: list[TritonTensorSpec]
    outputs: list[TritonTensorSpec]
    instance_count: int = 1
    instance_kind: TritonInstanceKind = TritonInstanceKind.KIND_GPU

    def to_pbtxt(self) -> str:
        """Render this configuration as Triton config.pbtxt protobuf text."""
        lines: list[str] = [
            f'name: "{self.name}"',
            f'platform: "{self.platform}"',
            f"max_batch_size: {self.max_batch_size}",
            "",
        ]

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
