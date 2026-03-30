"""Pydantic models for Triton model configuration and YOLO model registry."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class TritonDataType(StrEnum):
    """Triton Inference Server data types."""

    TYPE_FP32 = "TYPE_FP32"
    TYPE_FP16 = "TYPE_FP16"
    TYPE_INT8 = "TYPE_INT8"
    TYPE_INT16 = "TYPE_INT16"
    TYPE_INT32 = "TYPE_INT32"
    TYPE_INT64 = "TYPE_INT64"
    TYPE_UINT8 = "TYPE_UINT8"
    TYPE_BOOL = "TYPE_BOOL"


class TritonInstanceKind(StrEnum):
    """Triton instance group kind."""

    KIND_GPU = "KIND_GPU"
    KIND_CPU = "KIND_CPU"


class TritonTensorSpec(BaseModel):
    """Specification for a single Triton input or output tensor."""

    name: str
    data_type: TritonDataType
    dims: list[int]


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
            lines.extend(self._render_tensor("input", tensor))
            lines.append("")

        for tensor in self.outputs:
            lines.extend(self._render_tensor("output", tensor))
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

    @staticmethod
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


class YoloModelVariant(BaseModel):
    """Describes a single YOLO model variant to export for Triton."""

    name: str
    ultralytics_model: str
    export_kwargs: dict[str, Any] = {}
    imgsz: int = 640


YOLO_MODELS: tuple[YoloModelVariant, ...] = (
    YoloModelVariant(
        name="yolo26s",
        ultralytics_model="yolo26s.pt",
        export_kwargs={"end2end": True},
    ),
    YoloModelVariant(
        name="yolo11s",
        ultralytics_model="yolo11s.pt",
    ),
    YoloModelVariant(
        name="yolo12s",
        ultralytics_model="yolo12s.pt",
    ),
)
