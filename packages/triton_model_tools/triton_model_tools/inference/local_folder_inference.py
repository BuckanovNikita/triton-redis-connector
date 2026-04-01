"""Folder-level local ONNX inference using the same pre/postprocessing as Triton pipeline."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

from triton_model_tools.inference.detection_io import collect_image_paths, load_image
from triton_model_tools.inference.onnx_inference import OnnxInferenceClient
from triton_model_tools.inference.postprocessing import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    YoloPostprocessor,
)
from triton_model_tools.inference.preprocessing import YoloPreprocessor
from triton_model_tools.models.detection import FolderInferenceResult, ImageDetectionResult

MILLISECONDS_PER_SECOND = 1000.0

if TYPE_CHECKING:
    from pathlib import Path


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

    Uses the same YoloPreprocessor and YoloPostprocessor as the Triton pipeline,
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

    preprocessor = YoloPreprocessor(target_size=target_size)
    postprocessor = YoloPostprocessor(end2end=is_end2end)
    image_paths = collect_image_paths(image_dir)

    model_name = model_path.stem

    results: list[ImageDetectionResult] = []
    total_start = time.perf_counter()

    for image_path in image_paths:
        logger.info("Processing {}", image_path.name)
        image = load_image(image_path)

        img_start = time.perf_counter()

        tensor, scale, pad_offset = preprocessor.preprocess(image)
        outputs = client.infer(tensor)
        detections = postprocessor.postprocess(
            outputs,
            scale,
            pad_offset,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        img_time_ms = (time.perf_counter() - img_start) * MILLISECONDS_PER_SECOND

        results.append(
            ImageDetectionResult(
                image_path=str(image_path),
                model_name=model_name,
                detections=detections,
                inference_time_ms=img_time_ms,
            ),
        )
        logger.info(
            "{}: {} detections in {:.1f}ms",
            image_path.name,
            len(detections),
            img_time_ms,
        )

    total_time_s = time.perf_counter() - total_start
    avg_time_ms = (total_time_s * MILLISECONDS_PER_SECOND / len(results)) if results else 0.0

    logger.info(
        "Local ONNX inference complete: {} images, {:.2f}s total, {:.1f}ms avg",
        len(results),
        total_time_s,
        avg_time_ms,
    )

    return FolderInferenceResult(
        model_name=model_name,
        results=results,
        total_time_s=total_time_s,
        avg_time_ms=avg_time_ms,
    )
