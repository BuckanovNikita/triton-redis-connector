"""Folder-level batch inference using Triton and YOLO pre/postprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from triton_shared_models.preprocessing import LetterBoxPreprocessor as YoloPreprocessor

from triton_model_tools.inference._folder_inference_common import run_image_loop
from triton_model_tools.inference.client import TritonInferenceClient
from triton_model_tools.inference.detection_io import collect_image_paths
from triton_model_tools.inference.postprocessing import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    YoloPostprocessor,
)

if TYPE_CHECKING:
    from pathlib import Path

    from triton_model_tools.models.detection import FolderInferenceResult


def run_folder_inference(  # noqa: PLR0913
    image_dir: Path,
    model_name: str,
    *,
    triton_url: str = "localhost:8001",
    target_size: int | tuple[int, int] = 640,
    end2end: bool = False,
    output_names: list[str] | None = None,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> FolderInferenceResult:
    """Run YOLO inference on all images in a directory.

    Args:
        image_dir: Directory containing image files.
        model_name: Triton model name.
        triton_url: Triton gRPC URL.
        target_size: Input image size for the model.
        end2end: Whether model uses end2end NMS output format.
        output_names: Output tensor names. Defaults to ["output0"].
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for NMS.

    Returns:
        FolderInferenceResult with per-image results and timing.

    """
    resolved_output_names = output_names if output_names is not None else ["output0"]

    image_paths = collect_image_paths(image_dir)
    client = TritonInferenceClient(url=triton_url)
    preprocessor = YoloPreprocessor(target_size=target_size)
    postprocessor = YoloPostprocessor(end2end=end2end)

    return run_image_loop(
        image_paths=image_paths,
        model_name=model_name,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        infer_fn=lambda tensor: client.infer_raw(model_name, tensor, resolved_output_names),
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        log_label="Folder inference",
    )
