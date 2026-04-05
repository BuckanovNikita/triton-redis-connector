"""Folder-level local ONNX inference using the same pre/postprocessing as Triton pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from triton_shared_models.detection.preprocessing import LetterBoxPreprocessor

from triton_model_tools.inference._folder_inference_common import run_image_loop
from triton_model_tools.inference.detection_io import collect_image_paths
from triton_model_tools.inference.onnx_inference import OnnxInferenceClient
from triton_model_tools.inference.postprocessing import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    YoloPostprocessor,
)

if TYPE_CHECKING:
    from pathlib import Path

    from triton_model_tools.models.detection import FolderInferenceResult


def run_local_folder_inference(  # noqa: PLR0913
    model_path: Path,
    image_dir: Path,
    *,
    target_size: int | tuple[int, int] = 640,
    end2end: bool = False,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> FolderInferenceResult:
    """Run ONNX inference on all images in a directory.

    Uses the same LetterBoxPreprocessor and YoloPostprocessor as the Triton pipeline,
    but replaces the Triton gRPC client with a local OnnxInferenceClient.

    Args:
        model_path: Path to the .onnx model file.
        image_dir: Directory containing image files.
        target_size: Input image size for the model.
        end2end: Force end2end NMS output format. If False, auto-detected.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for NMS.

    Returns:
        FolderInferenceResult with per-image results and timing.

    """
    client = OnnxInferenceClient(model_path)

    is_end2end = end2end or client.detect_end2end()
    if is_end2end:
        logger.info("Model output format: end2end (NMS built-in)")
    else:
        logger.info("Model output format: standard (NMS applied in postprocessing)")

    preprocessor = LetterBoxPreprocessor(target_size=target_size)
    postprocessor = YoloPostprocessor(end2end=is_end2end)
    image_paths = collect_image_paths(image_dir)

    return run_image_loop(
        image_paths=image_paths,
        model_name=model_path.stem,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        infer_fn=client.infer,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        log_label="Local ONNX inference",
    )
