"""Core export logic: Ultralytics YOLO -> ONNX -> Triton model repository."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from triton_model_tools.models import (
    YOLO_MODELS,
    TritonDataType,
    TritonModelConfig,
    TritonTensorSpec,
)
from triton_model_tools.repository import build_model_directory

if TYPE_CHECKING:
    from collections.abc import Sequence

    import onnx

    from triton_model_tools.models import YoloModelVariant

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


def _extract_tensor_spec(
    tensor: onnx.TensorProto | onnx.ValueInfoProto,
) -> TritonTensorSpec:
    """Extract tensor name, dtype, and shape from an ONNX graph tensor."""
    import onnx  # noqa: PLC0415

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
    import onnx  # noqa: PLC0415

    model = onnx.load(str(onnx_path))  # pyright: ignore[reportUnknownMemberType]
    inputs = [_extract_tensor_spec(inp) for inp in model.graph.input]
    outputs = [_extract_tensor_spec(out) for out in model.graph.output]

    for spec in [*inputs, *outputs]:
        logger.debug("  {} {} {}", spec.name, spec.data_type, spec.dims)

    return inputs, outputs


def export_model(variant: YoloModelVariant, repository_root: Path) -> Path:
    """Export a single YOLO model to ONNX and build its Triton repository entry.

    Downloads the ultralytics .pt model (if not cached), exports it to ONNX,
    inspects the ONNX graph for real tensor shapes, and creates the Triton
    model directory with config.pbtxt.

    Args:
        variant: YOLO model variant specification.
        repository_root: Root path of the Triton model repository.

    Returns:
        Path to the created Triton model directory.

    """
    from ultralytics import YOLO  # type: ignore[attr-defined]  # noqa: PLC0415

    logger.info("Exporting {} from {}", variant.name, variant.ultralytics_model)

    model = YOLO(variant.ultralytics_model)
    onnx_path_str: str = model.export(
        format="onnx",
        imgsz=variant.imgsz,
        **variant.export_kwargs,
    )
    onnx_path = Path(onnx_path_str)
    logger.info("ONNX exported to {}", onnx_path)

    logger.info("Inspecting ONNX model tensors:")
    inputs, outputs = inspect_onnx_model(onnx_path)

    config = TritonModelConfig(
        name=variant.name,
        inputs=inputs,
        outputs=outputs,
    )

    model_dir = build_model_directory(repository_root, config, model_source=onnx_path)

    onnx_path.unlink()
    logger.info("Cleaned up intermediate ONNX file: {}", onnx_path)

    pt_path = Path(variant.ultralytics_model)
    if pt_path.exists():
        pt_path.unlink()
        logger.info("Cleaned up downloaded PT file: {}", pt_path)

    return model_dir


def export_all_models(
    repository_root: Path,
    model_names: Sequence[str] | None = None,
) -> list[Path]:
    """Export YOLO models to ONNX and build their Triton repository entries.

    Args:
        repository_root: Root path of the Triton model repository.
        model_names: Optional list of model names to export.
            If None, exports all registered models.

    Returns:
        List of paths to created Triton model directories.

    """
    variants = YOLO_MODELS
    if model_names is not None:
        name_set = set(model_names)
        variants = tuple(v for v in variants if v.name in name_set)
        unknown = name_set - {v.name for v in variants}
        if unknown:
            known = ", ".join(v.name for v in YOLO_MODELS)
            msg = f"Unknown model names: {', '.join(sorted(unknown))}. Available: {known}"
            raise ValueError(msg)

    repository_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Exporting {} model(s) to {}",
        len(variants),
        repository_root,
    )

    results: list[Path] = []
    for variant in variants:
        model_dir = export_model(variant, repository_root)
        results.append(model_dir)

    logger.info(
        "Export complete: {} model(s) ready in {}",
        len(results),
        repository_root,
    )
    return results
