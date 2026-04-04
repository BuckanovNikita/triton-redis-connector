"""Tests for detection comparison utilities."""

import pytest
from triton_model_tools.benchmarks.comparison import (
    compare_backends,
    compare_inference,
    match_detections,
)
from triton_model_tools.models.detection import Detection, ImageDetectionResult


def _det(  # noqa: PLR0913
    x1: float = 0.0,
    y1: float = 0.0,
    x2: float = 100.0,
    y2: float = 100.0,
    confidence: float = 0.9,
    class_id: int = 0,
) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence, class_id=class_id)


def _img_result(
    path: str = "img.jpg",
    detections: list[Detection] | None = None,
) -> ImageDetectionResult:
    return ImageDetectionResult(
        image_path=path,
        model_name="yolo11s",
        detections=detections or [],
        inference_time_ms=10.0,
    )


class TestMatchDetections:
    def test_perfect_match(self) -> None:
        preds = [_det(0, 0, 100, 100)]
        gts = [_det(0, 0, 100, 100)]
        matched, fp, missed = match_detections(preds, gts)
        assert len(matched) == 1
        assert len(fp) == 0
        assert len(missed) == 0

    def test_no_match_below_threshold(self) -> None:
        preds = [_det(0, 0, 50, 50)]
        gts = [_det(200, 200, 300, 300)]
        matched, fp, missed = match_detections(preds, gts)
        assert len(matched) == 0
        assert len(fp) == 1
        assert len(missed) == 1

    def test_empty_predictions(self) -> None:
        gts = [_det(0, 0, 100, 100)]
        matched, fp, missed = match_detections([], gts)
        assert len(matched) == 0
        assert len(fp) == 0
        assert len(missed) == 1

    def test_empty_ground_truth(self) -> None:
        preds = [_det(0, 0, 100, 100)]
        matched, fp, missed = match_detections(preds, [])
        assert len(matched) == 0
        assert len(fp) == 1
        assert len(missed) == 0

    def test_both_empty(self) -> None:
        matched, fp, missed = match_detections([], [])
        assert len(matched) == 0
        assert len(fp) == 0
        assert len(missed) == 0

    def test_multiple_matches(self) -> None:
        preds = [_det(0, 0, 100, 100), _det(200, 200, 300, 300)]
        gts = [_det(0, 0, 100, 100), _det(200, 200, 300, 300)]
        matched, fp, missed = match_detections(preds, gts)
        assert len(matched) == 2
        assert len(fp) == 0
        assert len(missed) == 0

    def test_more_predictions_than_gt(self) -> None:
        preds = [_det(0, 0, 100, 100), _det(200, 200, 300, 300), _det(400, 400, 500, 500)]
        gts = [_det(0, 0, 100, 100)]
        matched, fp, missed = match_detections(preds, gts)
        assert len(matched) == 1
        assert len(fp) == 2
        assert len(missed) == 0

    def test_more_gt_than_predictions(self) -> None:
        preds = [_det(0, 0, 100, 100)]
        gts = [_det(0, 0, 100, 100), _det(200, 200, 300, 300)]
        matched, fp, missed = match_detections(preds, gts)
        assert len(matched) == 1
        assert len(fp) == 0
        assert len(missed) == 1

    def test_custom_iou_threshold(self) -> None:
        a = _det(0, 0, 100, 100)
        b = _det(50, 50, 150, 150)
        matched_high, fp_high, missed_high = match_detections([a], [b], iou_threshold=0.5)
        assert len(matched_high) == 0
        assert len(fp_high) == 1
        assert len(missed_high) == 1

        matched_low, fp_low, missed_low = match_detections([a], [b], iou_threshold=0.1)
        assert len(matched_low) == 1
        assert len(fp_low) == 0
        assert len(missed_low) == 0

    def test_class_mismatch_still_matches_by_iou(self) -> None:
        pred = _det(0, 0, 100, 100, class_id=0)
        gt = _det(0, 0, 100, 100, class_id=5)
        matched, _fp, _missed = match_detections([pred], [gt])
        assert len(matched) == 1
        assert matched[0][0].class_id != matched[0][1].class_id


class TestCompareInference:
    def test_identical_results(self) -> None:
        dets = [_det(0, 0, 100, 100)]
        baseline = [_img_result("a.jpg", dets)]
        candidate = [_img_result("a.jpg", dets)]
        summary = compare_inference(baseline, candidate)
        assert summary.overall.precision == pytest.approx(1.0)
        assert summary.overall.recall == pytest.approx(1.0)
        assert summary.overall.avg_iou == pytest.approx(1.0)
        assert summary.overall.false_positive_count == 0
        assert summary.overall.missed_count == 0

    def test_no_overlap_results(self) -> None:
        baseline = [_img_result("a.jpg", [_det(0, 0, 50, 50)])]
        candidate = [_img_result("a.jpg", [_det(200, 200, 300, 300)])]
        summary = compare_inference(baseline, candidate)
        assert summary.overall.precision == pytest.approx(0.0)
        assert summary.overall.recall == pytest.approx(0.0)
        assert summary.overall.false_positive_count == 1
        assert summary.overall.missed_count == 1

    def test_empty_results(self) -> None:
        summary = compare_inference([], [])
        assert len(summary.per_image) == 0

    def test_missing_image_in_candidate(self) -> None:
        baseline = [_img_result("a.jpg", [_det()])]
        summary = compare_inference(baseline, [])
        assert summary.overall.missed_count == 1
        assert summary.overall.false_positive_count == 0

    def test_extra_image_in_candidate(self) -> None:
        candidate = [_img_result("a.jpg", [_det()])]
        summary = compare_inference([], candidate)
        assert summary.overall.false_positive_count == 1
        assert summary.overall.missed_count == 0

    def test_multi_image(self) -> None:
        baseline = [
            _img_result("a.jpg", [_det(0, 0, 100, 100)]),
            _img_result("b.jpg", [_det(0, 0, 50, 50)]),
        ]
        candidate = [
            _img_result("a.jpg", [_det(0, 0, 100, 100)]),
            _img_result("b.jpg", [_det(0, 0, 50, 50)]),
        ]
        summary = compare_inference(baseline, candidate)
        assert len(summary.per_image) == 2
        assert summary.overall.precision == pytest.approx(1.0)
        assert summary.overall.recall == pytest.approx(1.0)


class TestCompareBackends:
    def test_two_backends(self) -> None:
        dets = [_det(0, 0, 100, 100)]
        results = {
            "onnx": [_img_result("a.jpg", dets)],
            "tensorrt": [_img_result("a.jpg", dets)],
        }
        summaries = compare_backends(results, baseline_key="onnx")
        assert "tensorrt" in summaries
        assert "onnx" not in summaries
        assert summaries["tensorrt"].overall.precision == pytest.approx(1.0)

    def test_three_backends(self) -> None:
        dets = [_det(0, 0, 100, 100)]
        results = {
            "onnx": [_img_result("a.jpg", dets)],
            "tensorrt": [_img_result("a.jpg", dets)],
            "openvino": [_img_result("a.jpg", dets)],
        }
        summaries = compare_backends(results, baseline_key="onnx")
        assert len(summaries) == 2
        assert "tensorrt" in summaries
        assert "openvino" in summaries

    def test_divergent_results(self) -> None:
        results = {
            "onnx": [_img_result("a.jpg", [_det(0, 0, 100, 100)])],
            "bad": [_img_result("a.jpg", [_det(500, 500, 600, 600)])],
        }
        summaries = compare_backends(results, baseline_key="onnx")
        assert summaries["bad"].overall.precision == pytest.approx(0.0)
