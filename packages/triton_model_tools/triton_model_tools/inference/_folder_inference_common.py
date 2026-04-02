"""Shared image-loop logic for folder-level inference (Triton and local ONNX)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

from triton_model_tools.inference.detection_io import load_image
from triton_model_tools.models.constants import MILLISECONDS_PER_SECOND
from triton_model_tools.models.detection import FolderInferenceResult, ImageDetectionResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray
    from triton_shared_models.preprocessing import LetterBoxPreprocessor as YoloPreprocessor

    from triton_model_tools.inference.postprocessing import YoloPostprocessor


def run_image_loop(  # noqa: PLR0913
    *,
    image_paths: list[Path],
    model_name: str,
    preprocessor: YoloPreprocessor,
    postprocessor: YoloPostprocessor,
    infer_fn: Callable[[NDArray[Any]], dict[str, NDArray[Any]]],
    conf_threshold: float,
    iou_threshold: float,
    log_label: str = "Inference",
) -> FolderInferenceResult:
    """Run preprocessing → inference → postprocessing on each image and collect results.

    Args:
        image_paths: List of image file paths to process.
        model_name: Name used in result metadata.
        preprocessor: YOLO preprocessor (letterbox + normalize).
        postprocessor: YOLO postprocessor (NMS + filtering).
        infer_fn: Callable that takes a preprocessed tensor and returns output dict.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for NMS.
        log_label: Label for log messages.

    Returns:
        FolderInferenceResult with per-image results and timing.

    """
    results: list[ImageDetectionResult] = []
    total_start = time.perf_counter()

    for image_path in image_paths:
        logger.info("Processing {}", image_path.name)
        image = load_image(image_path)

        img_start = time.perf_counter()

        tensor, scale, pad_offset = preprocessor.preprocess(image)
        outputs = infer_fn(tensor)
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
        "{} complete: {} images, {:.2f}s total, {:.1f}ms avg",
        log_label,
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
