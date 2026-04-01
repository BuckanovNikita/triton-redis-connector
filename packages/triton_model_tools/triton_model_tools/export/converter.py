"""Convert YOLO .pt weights to Triton model repository format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

from loguru import logger

from triton_model_tools.export.exporter import export_model
from triton_model_tools.models.registry import YoloModelVariant


def convert_model(  # noqa: PLR0913
    weights_path: Path,
    repository_root: Path,
    *,
    name: str | None = None,
    imgsz: int | tuple[int, int] = 640,
    rect: bool = True,
    fmt: Literal["onnx", "tensorrt", "all"] = "onnx",
    half: bool = False,
    end2end: bool = False,
    export_kwargs: dict[str, Any] | None = None,
    enable_dynamic_batching: bool = False,
    max_batch_size: int = 8,
    trt_acceleration: bool = False,
) -> list[Path]:
    """Convert a YOLO .pt model to Triton repository format.

    Args:
        weights_path: Path to YOLO .pt weights file.
        repository_root: Root path of the Triton model repository.
        name: Model name in repository. Defaults to weights filename stem.
        imgsz: Input image size. Int for square, tuple (H, W) for rectangular.
        rect: Pass rect=True to ultralytics export for rectangular inference.
        fmt: Export format - "onnx", "tensorrt", or "all".
        half: Export in FP16 precision.
        end2end: Export with end-to-end NMS.
        export_kwargs: Additional kwargs for ultralytics export.
        enable_dynamic_batching: Enable Triton dynamic batching.
        max_batch_size: Maximum batch size for dynamic batching.
        trt_acceleration: Enable TensorRT execution accelerator for ONNX models.

    Returns:
        List of paths to created Triton model directories.

    Raises:
        FileNotFoundError: If weights file does not exist.

    """
    if not weights_path.exists():
        msg = f"Weights file not found: {weights_path}"
        raise FileNotFoundError(msg)

    model_name = name if name is not None else weights_path.stem

    merged_kwargs = dict(export_kwargs) if export_kwargs is not None else {}
    if end2end:
        merged_kwargs["end2end"] = True
    merged_kwargs["rect"] = rect

    variant = YoloModelVariant(
        name=model_name,
        ultralytics_model=str(weights_path),
        imgsz=imgsz,
        end2end_export=end2end,
        rect=rect,
        export_kwargs=merged_kwargs,
    )

    logger.info(
        "Converting {} -> {} (format={}, imgsz={}, rect={}, half={}, end2end={})",
        weights_path,
        model_name,
        fmt,
        imgsz,
        rect,
        half,
        end2end,
    )

    repository_root.mkdir(parents=True, exist_ok=True)

    return export_model(
        variant,
        repository_root,
        fmt=fmt,
        half=half,
        enable_dynamic_batching=enable_dynamic_batching,
        max_batch_size=max_batch_size,
        trt_acceleration=trt_acceleration,
    )
