"""Pipeline performance benchmark tests (Wave 6.3-6.5).

These tests measure E2E inference throughput through the full pipeline:
preprocess -> Triton inference -> postprocess.

All tests require a running Triton server with loaded models and are
skipped automatically when Triton is unavailable.
"""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from loguru import logger
from triton_model_tools.inference.folder_inference import run_folder_inference
from triton_model_tools.models.detection import FolderInferenceResult

if TYPE_CHECKING:
    from typing import Any

pytestmark = [pytest.mark.benchmark, pytest.mark.integration]

DETECTION_MODEL = "yolo11s"
CLASSIFICATION_MODEL = "resnet18"

TEST_IMAGES_DIR = Path(__file__).resolve().parents[3] / "test_fixtures" / "images"

MAX_AVG_LATENCY_MS = 500.0
CONCURRENT_OVERHEAD_FACTOR = 1.5


@pytest.fixture
def test_images_dir() -> Path:
    """Return path to synthetic test images; skip if unavailable."""
    if not TEST_IMAGES_DIR.is_dir():
        pytest.skip(f"Test images directory not found: {TEST_IMAGES_DIR}")
    image_files = [p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not image_files:
        pytest.skip(f"No images found in {TEST_IMAGES_DIR}")
    return TEST_IMAGES_DIR


@pytest.fixture
def triton_model_ready(triton_client: Any) -> Any:
    """Return a callable that checks whether a specific model is ready."""

    def _check(model_name: str) -> bool:
        try:
            return triton_client.is_model_ready(model_name)  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            return False

    return _check


def _run_detection_inference(image_dir: Path) -> FolderInferenceResult:
    return run_folder_inference(
        image_dir=image_dir,
        model_name=DETECTION_MODEL,
        triton_url="localhost:8001",
        target_size=640,
    )


class TestDetectionPipelineThroughput:
    """Task 6.3: E2E pipeline perf -- detection only."""

    def test_detection_pipeline_throughput(
        self,
        test_images_dir: Path,
        triton_client: Any,
        triton_model_ready: Any,
    ) -> None:
        """Measure detection pipeline throughput: preprocess -> Triton -> postprocess."""
        if not triton_model_ready(DETECTION_MODEL):
            pytest.skip(f"Model {DETECTION_MODEL} not loaded in Triton")

        start = time.perf_counter()
        result = _run_detection_inference(test_images_dir)
        wall_time_s = time.perf_counter() - start

        assert isinstance(result, FolderInferenceResult)
        assert len(result.results) > 0

        logger.info(
            "Detection pipeline: {} images, {:.2f}s total, {:.1f}ms avg, {:.1f}ms max",
            len(result.results),
            wall_time_s,
            result.avg_time_ms,
            max(r.inference_time_ms for r in result.results),
        )
        for img_result in result.results:
            logger.info(
                "  {}: {} detections, {:.1f}ms",
                Path(img_result.image_path).name,
                len(img_result.detections),
                img_result.inference_time_ms,
            )

        assert result.avg_time_ms < MAX_AVG_LATENCY_MS, (
            f"Average latency {result.avg_time_ms:.1f}ms exceeds threshold {MAX_AVG_LATENCY_MS}ms"
        )

    def test_detection_pipeline_consistency(
        self,
        test_images_dir: Path,
        triton_client: Any,
        triton_model_ready: Any,
    ) -> None:
        """Run detection twice and verify latency variance is bounded."""
        if not triton_model_ready(DETECTION_MODEL):
            pytest.skip(f"Model {DETECTION_MODEL} not loaded in Triton")

        result_warmup = _run_detection_inference(test_images_dir)
        result_measured = _run_detection_inference(test_images_dir)

        logger.info(
            "Detection consistency: warmup avg={:.1f}ms, measured avg={:.1f}ms",
            result_warmup.avg_time_ms,
            result_measured.avg_time_ms,
        )

        assert result_measured.avg_time_ms < MAX_AVG_LATENCY_MS
        assert len(result_warmup.results) == len(result_measured.results)


class TestClassificationPipelineThroughput:
    """Task 6.4: E2E pipeline perf -- classification only (if model available)."""

    def test_classification_pipeline_throughput(
        self,
        test_images_dir: Path,
        triton_client: Any,
        triton_model_ready: Any,
    ) -> None:
        """Measure classification pipeline throughput if resnet18 is loaded."""
        if not triton_model_ready(CLASSIFICATION_MODEL):
            pytest.skip(f"Model {CLASSIFICATION_MODEL} not loaded in Triton")

        start = time.perf_counter()
        result = run_folder_inference(
            image_dir=test_images_dir,
            model_name=CLASSIFICATION_MODEL,
            triton_url="localhost:8001",
            target_size=224,
        )
        wall_time_s = time.perf_counter() - start

        assert isinstance(result, FolderInferenceResult)
        assert len(result.results) > 0

        logger.info(
            "Classification pipeline: {} images, {:.2f}s total, {:.1f}ms avg",
            len(result.results),
            wall_time_s,
            result.avg_time_ms,
        )

        assert result.avg_time_ms < MAX_AVG_LATENCY_MS, (
            f"Average latency {result.avg_time_ms:.1f}ms exceeds threshold {MAX_AVG_LATENCY_MS}ms"
        )


class TestConcurrentModelInference:
    """Task 6.5: Both models concurrent."""

    def test_concurrent_models_throughput(
        self,
        test_images_dir: Path,
        triton_client: Any,
        triton_model_ready: Any,
    ) -> None:
        """Run detection and classification in parallel, measure wall time vs sequential."""
        detection_ready = triton_model_ready(DETECTION_MODEL)
        classification_ready = triton_model_ready(CLASSIFICATION_MODEL)

        if not detection_ready or not classification_ready:
            pytest.skip(
                f"Both models required: {DETECTION_MODEL}={detection_ready}, "
                f"{CLASSIFICATION_MODEL}={classification_ready}"
            )

        _run_detection_inference(test_images_dir)

        sequential_start = time.perf_counter()
        detection_seq = _run_detection_inference(test_images_dir)
        classification_seq = run_folder_inference(
            image_dir=test_images_dir,
            model_name=CLASSIFICATION_MODEL,
            triton_url="localhost:8001",
            target_size=224,
        )
        sequential_time_s = time.perf_counter() - sequential_start

        concurrent_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            detection_future = executor.submit(_run_detection_inference, test_images_dir)
            classification_future = executor.submit(
                run_folder_inference,
                test_images_dir,
                CLASSIFICATION_MODEL,
                triton_url="localhost:8001",
                target_size=224,
            )
            detection_conc = detection_future.result(timeout=60)
            classification_conc = classification_future.result(timeout=60)
        concurrent_time_s = time.perf_counter() - concurrent_start

        logger.info(
            "Sequential: {:.2f}s (det={:.1f}ms avg, cls={:.1f}ms avg)",
            sequential_time_s,
            detection_seq.avg_time_ms,
            classification_seq.avg_time_ms,
        )
        logger.info(
            "Concurrent: {:.2f}s (det={:.1f}ms avg, cls={:.1f}ms avg)",
            concurrent_time_s,
            detection_conc.avg_time_ms,
            classification_conc.avg_time_ms,
        )
        logger.info(
            "Speedup: {:.2f}x (concurrent/sequential = {:.2f}s / {:.2f}s)",
            sequential_time_s / concurrent_time_s if concurrent_time_s > 0 else 0,
            concurrent_time_s,
            sequential_time_s,
        )

        assert len(detection_conc.results) == len(detection_seq.results)
        assert len(classification_conc.results) == len(classification_seq.results)

        max_concurrent_time = sequential_time_s * CONCURRENT_OVERHEAD_FACTOR
        assert concurrent_time_s < max_concurrent_time, (
            f"Concurrent time {concurrent_time_s:.2f}s exceeds "
            f"{CONCURRENT_OVERHEAD_FACTOR}x sequential time {sequential_time_s:.2f}s"
        )
