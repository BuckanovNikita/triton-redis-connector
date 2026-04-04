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

from pathlib import Path

import pytest
from triton_model_tools.benchmarks.comparison import compare_inference

MIN_PRECISION = 0.99
MIN_RECALL = 0.99
MIN_AVG_IOU = 0.99


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
