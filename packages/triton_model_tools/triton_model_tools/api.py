"""Public Python API for triton-model-tools.

Provides convenient functions for detection, classification, and model comparison
without needing to know internal module structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from triton_model_tools.benchmarks.comparison import (
        ClassificationComparisonSummary,
        ComparisonSummary,
    )
    from triton_model_tools.models.classification import ImageClassificationResult
    from triton_model_tools.models.detection import (
        FolderInferenceResult,
        ImageDetectionResult,
    )


def run_detection(  # noqa: PLR0913
    image_dir: Path,
    model_name: str,
    *,
    triton_url: str = "localhost:8001",
    target_size: int | tuple[int, int] = 640,
    end2end: bool = False,
    output_names: list[str] | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> FolderInferenceResult:
    """Run YOLO detection on all images in a directory via Triton.

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
    from triton_model_tools.inference.folder_inference import run_folder_inference

    return run_folder_inference(
        image_dir=image_dir,
        model_name=model_name,
        triton_url=triton_url,
        target_size=target_size,
        end2end=end2end,
        output_names=output_names,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


def run_local_detection(  # noqa: PLR0913
    model_path: Path,
    image_dir: Path,
    *,
    target_size: int | tuple[int, int] = 640,
    end2end: bool = False,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> FolderInferenceResult:
    """Run YOLO detection locally via ONNX runtime (no Triton needed).

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
    from triton_model_tools.inference.local_folder_inference import (
        run_local_folder_inference,
    )

    return run_local_folder_inference(
        model_path=model_path,
        image_dir=image_dir,
        target_size=target_size,
        end2end=end2end,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


def compare_detection_results(
    baseline: list[ImageDetectionResult],
    candidate: list[ImageDetectionResult],
    iou_threshold: float = 0.5,
) -> ComparisonSummary:
    """Compare candidate detections against baseline detections.

    Matches images by image_path. Images present in one set but not the
    other contribute as fully missed or fully false-positive.

    Args:
        baseline: Reference detection results.
        candidate: Detection results to evaluate.
        iou_threshold: IoU threshold for matching detections.

    Returns:
        ComparisonSummary with per-image and overall metrics.

    """
    from triton_model_tools.benchmarks.comparison import compare_inference

    return compare_inference(baseline, candidate, iou_threshold)


def compare_classification_results(
    baseline: list[ImageClassificationResult],
    candidate: list[ImageClassificationResult],
) -> ClassificationComparisonSummary:
    """Compare candidate classification results against baseline.

    Matches images by image_path. For each matched pair, checks if top-1
    prediction class_id matches and if baseline top-1 appears in candidate top-5.

    Args:
        baseline: Reference classification results.
        candidate: Classification results to evaluate.

    Returns:
        ClassificationComparisonSummary with per-image and overall metrics.

    """
    from triton_model_tools.benchmarks.comparison import compare_classification

    return compare_classification(baseline, candidate)
