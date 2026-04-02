"""Drift tests: detect broken models via comparison utilities.

Task 3.10 — detection drift with corrupted results.
Task 3.11 — classification drift with corrupted results.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from triton_model_tools.benchmarks.comparison import (
    ComparisonSummary,
    compare_classification,
    compare_inference,
)
from triton_model_tools.models.classification import (
    ClassificationPrediction,
    ImageClassificationResult,
)
from triton_model_tools.models.detection import Detection, ImageDetectionResult

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "test_fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"

IMAGE_NAMES = [f"test_{i:04d}.jpg" for i in range(8)]
SYNTHETIC_SEED = 42


def _make_realistic_detections(seed: int) -> list[Detection]:
    """Generate deterministic pseudo-realistic detections for one image."""
    rng = random.Random(seed)
    num_dets = rng.randint(1, 5)
    detections: list[Detection] = []
    for _ in range(num_dets):
        x1 = rng.uniform(0, 400)
        y1 = rng.uniform(0, 400)
        w = rng.uniform(30, 200)
        h = rng.uniform(30, 200)
        detections.append(
            Detection(
                x1=x1,
                y1=y1,
                x2=x1 + w,
                y2=y1 + h,
                confidence=rng.uniform(0.5, 0.99),
                class_id=rng.randint(0, 79),
            ),
        )
    return detections


def _build_baseline_detection_results() -> list[ImageDetectionResult]:
    """Build a set of baseline detection results with deterministic seed."""
    results: list[ImageDetectionResult] = []
    for idx, name in enumerate(IMAGE_NAMES):
        dets = _make_realistic_detections(seed=SYNTHETIC_SEED + idx)
        results.append(
            ImageDetectionResult(
                image_path=name,
                model_name="yolo11s",
                detections=dets,
                inference_time_ms=10.0,
            ),
        )
    return results


def _build_corrupted_zero_detections() -> list[ImageDetectionResult]:
    """Corrupted model: zero detections for every image."""
    return [
        ImageDetectionResult(
            image_path=name,
            model_name="yolo11s-corrupted",
            detections=[],
            inference_time_ms=10.0,
        )
        for name in IMAGE_NAMES
    ]


def _build_corrupted_random_detections() -> list[ImageDetectionResult]:
    """Corrupted model: random bboxes unrelated to real objects."""
    rng = random.Random(999)
    results: list[ImageDetectionResult] = []
    for name in IMAGE_NAMES:
        dets: list[Detection] = []
        for _ in range(rng.randint(1, 8)):
            x1 = rng.uniform(500, 900)
            y1 = rng.uniform(500, 900)
            dets.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x1 + rng.uniform(10, 50),
                    y2=y1 + rng.uniform(10, 50),
                    confidence=rng.uniform(0.1, 0.5),
                    class_id=rng.randint(0, 79),
                ),
            )
        results.append(
            ImageDetectionResult(
                image_path=name,
                model_name="yolo11s-corrupted",
                detections=dets,
                inference_time_ms=10.0,
            ),
        )
    return results


class TestDetectionDriftWithZeroDetections:
    """Task 3.10: Corrupted model producing zero detections must be caught."""

    def test_precision_drops_when_model_returns_nothing(self) -> None:
        baseline = _build_baseline_detection_results()
        corrupted = _build_corrupted_zero_detections()

        summary: ComparisonSummary = compare_inference(baseline, corrupted)

        assert summary.overall.recall == pytest.approx(0.0)
        total_baseline_dets = sum(len(r.detections) for r in baseline)
        assert summary.overall.missed_count == total_baseline_dets
        assert summary.overall.false_positive_count == 0

    def test_every_image_has_zero_recall(self) -> None:
        baseline = _build_baseline_detection_results()
        corrupted = _build_corrupted_zero_detections()

        summary = compare_inference(baseline, corrupted)

        for per_img in summary.per_image:
            assert per_img.recall == pytest.approx(0.0)


class TestDetectionDriftWithRandomBboxes:
    """Task 3.10: Corrupted model producing random bboxes must have low precision/recall."""

    def test_precision_and_recall_drop_significantly(self) -> None:
        baseline = _build_baseline_detection_results()
        corrupted = _build_corrupted_random_detections()

        summary = compare_inference(baseline, corrupted)

        assert summary.overall.precision < 0.5
        assert summary.overall.recall < 0.5

    def test_false_positives_and_misses_are_nonzero(self) -> None:
        baseline = _build_baseline_detection_results()
        corrupted = _build_corrupted_random_detections()

        summary = compare_inference(baseline, corrupted)

        assert summary.overall.false_positive_count > 0
        assert summary.overall.missed_count > 0


def _build_baseline_classification_results() -> list[ImageClassificationResult]:
    """Build baseline classification results with deterministic classes."""
    rng = random.Random(SYNTHETIC_SEED)
    results: list[ImageClassificationResult] = []
    for name in IMAGE_NAMES:
        top_class = rng.randint(0, 999)
        preds = [
            ClassificationPrediction(
                class_id=top_class,
                probability=0.85,
            ),
            ClassificationPrediction(
                class_id=(top_class + 1) % 1000,
                probability=0.10,
            ),
            ClassificationPrediction(
                class_id=(top_class + 2) % 1000,
                probability=0.03,
            ),
            ClassificationPrediction(
                class_id=(top_class + 3) % 1000,
                probability=0.01,
            ),
            ClassificationPrediction(
                class_id=(top_class + 4) % 1000,
                probability=0.01,
            ),
        ]
        results.append(
            ImageClassificationResult(
                image_path=name,
                model_name="resnet18",
                predictions=preds,
                inference_time_ms=5.0,
            ),
        )
    return results


def _build_corrupted_classification_results() -> list[ImageClassificationResult]:
    """Corrupted model: wrong class IDs unrelated to baseline."""
    baseline_rng = random.Random(SYNTHETIC_SEED)
    results: list[ImageClassificationResult] = []
    for name in IMAGE_NAMES:
        real_top = baseline_rng.randint(0, 999)
        wrong_class = (real_top + 500) % 1000
        preds = [
            ClassificationPrediction(
                class_id=wrong_class,
                probability=0.7,
            ),
            ClassificationPrediction(
                class_id=(wrong_class + 100) % 1000,
                probability=0.15,
            ),
            ClassificationPrediction(
                class_id=(wrong_class + 200) % 1000,
                probability=0.08,
            ),
            ClassificationPrediction(
                class_id=(wrong_class + 300) % 1000,
                probability=0.05,
            ),
            ClassificationPrediction(
                class_id=(wrong_class + 400) % 1000,
                probability=0.02,
            ),
        ]
        results.append(
            ImageClassificationResult(
                image_path=name,
                model_name="resnet18-corrupted",
                predictions=preds,
                inference_time_ms=5.0,
            ),
        )
    return results


class TestClassificationDrift:
    """Task 3.11: Corrupted classification model must show low accuracy."""

    def test_top1_accuracy_drops_with_wrong_classes(self) -> None:
        baseline = _build_baseline_classification_results()
        corrupted = _build_corrupted_classification_results()

        summary = compare_classification(baseline, corrupted)

        assert summary.overall.top1_accuracy < 0.5
        assert summary.overall.class_mismatch_rate > 0.5

    def test_top5_accuracy_drops_with_wrong_classes(self) -> None:
        baseline = _build_baseline_classification_results()
        corrupted = _build_corrupted_classification_results()

        summary = compare_classification(baseline, corrupted)

        assert summary.overall.top5_accuracy < 0.5

    def test_per_image_all_mismatch(self) -> None:
        baseline = _build_baseline_classification_results()
        corrupted = _build_corrupted_classification_results()

        summary = compare_classification(baseline, corrupted)

        for per_img in summary.per_image:
            assert per_img.top1_accuracy == pytest.approx(0.0)

    def test_identical_classification_has_perfect_accuracy(self) -> None:
        baseline = _build_baseline_classification_results()

        summary = compare_classification(baseline, baseline)

        assert summary.overall.top1_accuracy == pytest.approx(1.0)
        assert summary.overall.top5_accuracy == pytest.approx(1.0)
        assert summary.overall.class_mismatch_rate == pytest.approx(0.0)
