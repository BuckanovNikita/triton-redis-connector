"""Core export logic: Ultralytics YOLO -> ONNX/TensorRT -> Triton model repository."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import onnx
from loguru import logger
from ultralytics import YOLO  # type: ignore[attr-defined]

from triton_model_tools.export.repository import build_model_directory
from triton_model_tools.models import (
    EXPORT_FORMAT_TO_FILENAME,
    EXPORT_FORMAT_TO_PLATFORM,
    DynamicBatchingConfig,
    ExportFormat,
    TritonDataType,
    TritonModelConfig,
    TritonTensorSpec,
)

if TYPE_CHECKING:
    from triton_model_tools.models.registry import YoloModelVariant

ONNX_DTYPE_TO_TRITON: dict[int, TritonDataType] = {
    1: TritonDataType.TYPE_FP32,
    10: TritonDataType.TYPE_FP16,
    2: TritonDataType.TYPE_UINT8,
    3: TritonDataType.TYPE_INT8,
    5: TritonDataType.TYPE_INT16,
    6: TritonDataType.TYPE_INT32,
    7: TritonDataType.TYPE_INT64,
    9: TritonDataType.TYPE_BOOL,
}

TENSORRT_NAME_SUFFIX = "_trt"


def _extract_tensor_spec(
    tensor: onnx.TensorProto | onnx.ValueInfoProto,
) -> TritonTensorSpec:
    """Extract tensor name, dtype, and shape from an ONNX graph tensor."""
    name = tensor.name

    if isinstance(tensor, onnx.TensorProto):
        data_type = ONNX_DTYPE_TO_TRITON.get(tensor.data_type, TritonDataType.TYPE_FP32)
        dims = list(tensor.dims)
    else:
        type_proto = tensor.type.tensor_type
        data_type = ONNX_DTYPE_TO_TRITON.get(type_proto.elem_type, TritonDataType.TYPE_FP32)
        dims = [dim.dim_value if dim.dim_value > 0 else -1 for dim in type_proto.shape.dim]

    return TritonTensorSpec(name=name, data_type=data_type, dims=dims)


def inspect_onnx_model(
    onnx_path: Path,
) -> tuple[list[TritonTensorSpec], list[TritonTensorSpec]]:
    """Inspect an ONNX model file and extract input/output tensor specifications.

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        Tuple of (input_specs, output_specs) extracted from the ONNX graph.

    """
    model = onnx.load(str(onnx_path))
    inputs = [_extract_tensor_spec(inp) for inp in model.graph.input]
    outputs = [_extract_tensor_spec(out) for out in model.graph.output]

    for spec in [*inputs, *outputs]:
        logger.debug("  {} {} {}", spec.name, spec.data_type, spec.dims)

    return inputs, outputs


def _strip_batch_dim(specs: list[TritonTensorSpec]) -> list[TritonTensorSpec]:
    """Remove the leading batch dimension from tensor specs for dynamic batching."""
    return [
        TritonTensorSpec(name=spec.name, data_type=spec.data_type, dims=spec.dims[1:])
        for spec in specs
        if len(spec.dims) > 1
    ]


def _export_onnx(
    variant: YoloModelVariant,
    repository_root: Path,
    *,
    half: bool = False,
    enable_dynamic_batching: bool = False,
    max_batch_size: int = 8,
) -> Path:
    """Export a YOLO model to ONNX and build its Triton repository entry.

    Args:
        variant: YOLO model variant specification.
        repository_root: Root path of the Triton model repository.
        half: If True, export ONNX model in FP16 precision.
        enable_dynamic_batching: If True, configure Triton dynamic batching.
        max_batch_size: Maximum batch size when dynamic batching is enabled.

    Returns:
        Path to the created Triton model directory.

    """
    logger.info("Exporting {} (ONNX, half={}) from {}", variant.name, half, variant.ultralytics_model)

    export_kwargs = dict(variant.export_kwargs)
    if enable_dynamic_batching:
        export_kwargs["dynamic"] = True

    yolo_model = YOLO(variant.ultralytics_model)
    onnx_path_str: str = yolo_model.export(
        format="onnx",
        imgsz=list(variant.resolved_imgsz),
        half=half,
        **export_kwargs,
    )
    onnx_path = Path(onnx_path_str)
    logger.info("ONNX exported to {}", onnx_path)

    logger.info("Inspecting ONNX model tensors:")
    inputs, outputs = inspect_onnx_model(onnx_path)

    dynamic_batching: DynamicBatchingConfig | None = None
    effective_max_batch_size = 0
    if enable_dynamic_batching:
        inputs = _strip_batch_dim(inputs)
        outputs = _strip_batch_dim(outputs)
        effective_max_batch_size = max_batch_size
        dynamic_batching = DynamicBatchingConfig()
        logger.info("Dynamic batching enabled: max_batch_size={}", max_batch_size)

    config = TritonModelConfig(
        name=variant.name,
        platform=EXPORT_FORMAT_TO_PLATFORM[ExportFormat.ONNX],
        inputs=inputs,
        outputs=outputs,
        max_batch_size=effective_max_batch_size,
        dynamic_batching=dynamic_batching,
    )

    model_dir = build_model_directory(
        repository_root,
        config,
        model_source=onnx_path,
        model_filename=EXPORT_FORMAT_TO_FILENAME[ExportFormat.ONNX],
    )

    onnx_path.unlink()
    logger.info("Cleaned up intermediate ONNX file: {}", onnx_path)

    pt_path = Path(variant.ultralytics_model)
    if pt_path.exists():
        pt_path.unlink()
        logger.info("Cleaned up downloaded PT file: {}", pt_path)

    return model_dir


def _export_tensorrt(
    variant: YoloModelVariant,
    repository_root: Path,
    *,
    half: bool = True,
    enable_dynamic_batching: bool = False,
    max_batch_size: int = 8,
) -> Path:
    """Export a YOLO model to TensorRT engine and build its Triton repository entry.

    Uses FP16 precision by default for optimal GPU inference performance.

    Args:
        variant: YOLO model variant specification.
        repository_root: Root path of the Triton model repository.
        half: If True (default), export TensorRT engine in FP16 precision.
        enable_dynamic_batching: If True, configure Triton dynamic batching.
        max_batch_size: Maximum batch size when dynamic batching is enabled.

    Returns:
        Path to the created Triton model directory.

    """
    trt_model_name = f"{variant.name}{TENSORRT_NAME_SUFFIX}"
    logger.info("Exporting {} (TensorRT, half={}) from {}", trt_model_name, half, variant.ultralytics_model)

    export_kwargs = dict(variant.export_kwargs)
    if enable_dynamic_batching:
        export_kwargs["dynamic"] = True

    yolo_model = YOLO(variant.ultralytics_model)
    engine_path_str: str = yolo_model.export(
        format="engine",
        imgsz=list(variant.resolved_imgsz),
        half=half,
        **export_kwargs,
    )
    engine_path = Path(engine_path_str)
    logger.info("TensorRT engine exported to {}", engine_path)

    trt_dtype = TritonDataType.TYPE_FP16 if half else TritonDataType.TYPE_FP32
    resolved_h, resolved_w = variant.resolved_imgsz

    dynamic_batching: DynamicBatchingConfig | None = None
    effective_max_batch_size = 0

    if enable_dynamic_batching:
        inputs = [
            TritonTensorSpec(
                name="images",
                data_type=trt_dtype,
                dims=[3, resolved_h, resolved_w],
            ),
        ]
        output_dims = [-1, 6] if variant.end2end_export else [-1, -1]
        outputs = [
            TritonTensorSpec(
                name="output0",
                data_type=trt_dtype,
                dims=output_dims,
            ),
        ]
        effective_max_batch_size = max_batch_size
        dynamic_batching = DynamicBatchingConfig()
        logger.info("Dynamic batching enabled: max_batch_size={}", max_batch_size)
    else:
        inputs = [
            TritonTensorSpec(
                name="images",
                data_type=trt_dtype,
                dims=[1, 3, resolved_h, resolved_w],
            ),
        ]
        output_dims = [1, -1, 6] if variant.end2end_export else [-1, -1]
        outputs = [
            TritonTensorSpec(
                name="output0",
                data_type=trt_dtype,
                dims=output_dims,
            ),
        ]

    config = TritonModelConfig(
        name=trt_model_name,
        platform=EXPORT_FORMAT_TO_PLATFORM[ExportFormat.TENSORRT],
        inputs=inputs,
        outputs=outputs,
        max_batch_size=effective_max_batch_size,
        dynamic_batching=dynamic_batching,
    )

    model_dir = build_model_directory(
        repository_root,
        config,
        model_source=engine_path,
        model_filename=EXPORT_FORMAT_TO_FILENAME[ExportFormat.TENSORRT],
    )

    engine_path.unlink()
    logger.info("Cleaned up intermediate TensorRT engine file: {}", engine_path)

    pt_path = Path(variant.ultralytics_model)
    if pt_path.exists():
        pt_path.unlink()
        logger.info("Cleaned up downloaded PT file: {}", pt_path)

    return model_dir


def export_model(  # noqa: PLR0913
    variant: YoloModelVariant,
    repository_root: Path,
    fmt: Literal["onnx", "tensorrt", "all"] = "onnx",
    *,
    half: bool = False,
    enable_dynamic_batching: bool = False,
    max_batch_size: int = 8,
) -> list[Path]:
    """Export a single YOLO model in the requested format(s) and build Triton entries.

    Args:
        variant: YOLO model variant specification.
        repository_root: Root path of the Triton model repository.
        fmt: Export format - "onnx", "tensorrt", or "all" for both.
        half: If True, export in FP16 precision. For ONNX defaults to FP32,
            for TensorRT defaults to FP16 regardless of this flag.
        enable_dynamic_batching: If True, configure Triton dynamic batching.
        max_batch_size: Maximum batch size when dynamic batching is enabled.

    Returns:
        List of paths to created Triton model directories.

    """
    results: list[Path] = []
    if fmt in ("onnx", "all"):
        results.append(
            _export_onnx(
                variant,
                repository_root,
                half=half,
                enable_dynamic_batching=enable_dynamic_batching,
                max_batch_size=max_batch_size,
            )
        )
    if fmt in ("tensorrt", "all"):
        results.append(
            _export_tensorrt(
                variant,
                repository_root,
                half=half,
                enable_dynamic_batching=enable_dynamic_batching,
                max_batch_size=max_batch_size,
            )
        )
    return results
