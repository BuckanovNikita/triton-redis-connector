"""Dynamic batching drift verification tests.

Verifies that enabling dynamic batching on a YOLO model does not cause
detection drift beyond acceptable thresholds. Compares inference results
from a baseline (no dynamic batching) model variant against a dynamic
batching variant.

Requirements for integration tests:
- Triton server running with both model variants loaded
- COCO test images available
- Both model variants must be configured in Triton model repository
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from triton_model_tools.benchmarks.comparison import compare_inference
from triton_model_tools.models.detection import Detection, ImageDetectionResult

if TYPE_CHECKING:
    from pathlib import Path

MIN_PRECISION = 0.99
MIN_RECALL = 0.99
MIN_AVG_IOU = 0.99


def _make_detection(  # noqa: PLR0913
    x1: float = 10.0,
    y1: float = 20.0,
    x2: float = 100.0,
    y2: float = 200.0,
    confidence: float = 0.9,
    class_id: int = 0,
) -> Detection:
    return Detection(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
        class_id=class_id,
    )


def _make_result(
    image_path: str,
    detections: list[Detection],
) -> ImageDetectionResult:
    return ImageDetectionResult(
        image_path=image_path,
        model_name="yolo11s",
        detections=detections,
        inference_time_ms=10.0,
    )


class TestDynamicBatchingDriftUnit:
    """Unit tests for drift comparison logic using synthetic data.

    These validate the comparison framework works correctly for the
    dynamic batching use case, without requiring Triton infrastructure.
    """

    def test_identical_results_pass_drift_check(self) -> None:
        dets = [
            _make_detection(10, 20, 100, 200, 0.95, 0),
            _make_detection(300, 400, 500, 600, 0.87, 1),
        ]
        baseline = [
            _make_result("img1.jpg", dets),
            _make_result("img2.jpg", dets),
        ]
        candidate = [
            _make_result("img1.jpg", dets),
            _make_result("img2.jpg", dets),
        ]

        summary = compare_inference(baseline, candidate)

        assert summary.overall.precision >= MIN_PRECISION
        assert summary.overall.recall >= MIN_RECALL
        assert summary.overall.avg_iou >= MIN_AVG_IOU

    def test_minor_coordinate_drift_still_passes(self) -> None:
        baseline_dets = [_make_detection(10.0, 20.0, 100.0, 200.0, 0.95, 0)]
        candidate_dets = [_make_detection(10.1, 20.1, 100.1, 200.1, 0.949, 0)]

        baseline = [_make_result("img.jpg", baseline_dets)]
        candidate = [_make_result("img.jpg", candidate_dets)]

        summary = compare_inference(baseline, candidate)

        assert summary.overall.precision >= MIN_PRECISION
        assert summary.overall.recall >= MIN_RECALL
        assert summary.overall.avg_iou > 0.99

    def test_significant_drift_fails_check(self) -> None:
        baseline_dets = [_make_detection(10, 20, 100, 200, 0.95, 0)]
        candidate_dets = [_make_detection(300, 400, 500, 600, 0.80, 1)]

        baseline = [_make_result("img.jpg", baseline_dets)]
        candidate = [_make_result("img.jpg", candidate_dets)]

        summary = compare_inference(baseline, candidate)

        assert summary.overall.precision < MIN_PRECISION
        assert summary.overall.recall < MIN_RECALL

    def test_missing_detection_in_candidate(self) -> None:
        baseline_dets = [
            _make_detection(10, 20, 100, 200, 0.95, 0),
            _make_detection(300, 400, 500, 600, 0.87, 1),
        ]
        candidate_dets = [
            _make_detection(10, 20, 100, 200, 0.95, 0),
        ]

        baseline = [_make_result("img.jpg", baseline_dets)]
        candidate = [_make_result("img.jpg", candidate_dets)]

        summary = compare_inference(baseline, candidate)

        assert summary.overall.recall < MIN_RECALL

    def test_extra_detection_in_candidate_lowers_precision(self) -> None:
        baseline_dets = [_make_detection(10, 20, 100, 200, 0.95, 0)]
        candidate_dets = [
            _make_detection(10, 20, 100, 200, 0.95, 0),
            _make_detection(500, 500, 600, 600, 0.5, 2),
        ]

        baseline = [_make_result("img.jpg", baseline_dets)]
        candidate = [_make_result("img.jpg", candidate_dets)]

        summary = compare_inference(baseline, candidate)

        assert summary.overall.precision < MIN_PRECISION
        assert summary.overall.recall >= MIN_RECALL


@pytest.mark.integration
@pytest.mark.benchmark
class TestDynamicBatchingDriftIntegration:
    """Integration tests requiring Triton with both model variants loaded.

    The baseline model should be loaded without dynamic batching,
    and the candidate should have dynamic batching enabled.
    """

    def test_yolo11s_dynamic_batching_no_drift(
        self,
        triton_client: object,
        coco_image_dir: Path,
    ) -> None:
        from triton_model_tools.inference.folder_inference import run_folder_inference

        baseline_model = "yolo11s"
        batched_model = "yolo11s_dynamic"

        baseline_result = run_folder_inference(
            image_dir=coco_image_dir,
            model_name=baseline_model,
            triton_url="localhost:8001",
        )
        batched_result = run_folder_inference(
            image_dir=coco_image_dir,
            model_name=batched_model,
            triton_url="localhost:8001",
        )

        summary = compare_inference(
            baseline_result.results,
            batched_result.results,
        )

        assert summary.overall.precision >= MIN_PRECISION, (
            f"Precision {summary.overall.precision:.4f} below threshold {MIN_PRECISION}"
        )
        assert summary.overall.recall >= MIN_RECALL, f"Recall {summary.overall.recall:.4f} below threshold {MIN_RECALL}"
        assert summary.overall.avg_iou >= MIN_AVG_IOU, (
            f"Avg IoU {summary.overall.avg_iou:.4f} below threshold {MIN_AVG_IOU}"
        )
