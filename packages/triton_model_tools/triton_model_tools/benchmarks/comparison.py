"""Prediction comparison utilities for evaluating detection model outputs.

Provides IoU computation, detection matching with greedy algorithm,
and multi-backend comparison summaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from triton_model_tools.models.detection import Detection, ImageDetectionResult


class ComparisonMetrics(BaseModel):
    """Precision, recall, and IoU metrics for a set of matched detections."""

    model_config = ConfigDict(frozen=True)

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    avg_iou: float = Field(ge=0.0, le=1.0)
    class_mismatch_rate: float = Field(ge=0.0, le=1.0)
    false_positive_count: int = Field(ge=0)
    missed_count: int = Field(ge=0)


class ComparisonSummary(BaseModel):
    """Per-image and overall comparison metrics between two detection sets."""

    model_config = ConfigDict(frozen=True)

    per_image: list[ComparisonMetrics]
    overall: ComparisonMetrics


def compute_iou(box_a: Detection, box_b: Detection) -> float:
    """Compute Intersection over Union between two detection bounding boxes."""
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, box_a.x2 - box_a.x1) * max(0.0, box_a.y2 - box_a.y1)
    area_b = max(0.0, box_b.x2 - box_b.x1) * max(0.0, box_b.y2 - box_b.y1)
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def match_detections(
    predictions: list[Detection],
    ground_truth: list[Detection],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[Detection, Detection]], list[Detection], list[Detection]]:
    """Match predictions to ground truth using greedy IoU-based matching.

    Returns:
        Tuple of (matched_pairs, false_positives, missed_detections).
        matched_pairs: list of (prediction, ground_truth) pairs.
        false_positives: predictions with no matching ground truth.
        missed_detections: ground truth with no matching prediction.

    """
    if not predictions and not ground_truth:
        return [], [], []
    if not predictions:
        return [], [], list(ground_truth)
    if not ground_truth:
        return [], list(predictions), []

    iou_matrix: list[tuple[float, int, int]] = []
    for p_idx, pred in enumerate(predictions):
        for g_idx, gt in enumerate(ground_truth):
            iou = compute_iou(pred, gt)
            if iou >= iou_threshold:
                iou_matrix.append((iou, p_idx, g_idx))

    iou_matrix.sort(key=lambda x: x[0], reverse=True)

    matched_pred_indices: set[int] = set()
    matched_gt_indices: set[int] = set()
    matched_pairs: list[tuple[Detection, Detection]] = []

    for _iou_val, p_idx, g_idx in iou_matrix:
        if p_idx in matched_pred_indices or g_idx in matched_gt_indices:
            continue
        matched_pairs.append((predictions[p_idx], ground_truth[g_idx]))
        matched_pred_indices.add(p_idx)
        matched_gt_indices.add(g_idx)

    false_positives = [p for i, p in enumerate(predictions) if i not in matched_pred_indices]
    missed_detections = [g for i, g in enumerate(ground_truth) if i not in matched_gt_indices]

    return matched_pairs, false_positives, missed_detections


def _compute_metrics(
    matched_pairs: list[tuple[Detection, Detection]],
    false_positives: list[Detection],
    missed_detections: list[Detection],
) -> ComparisonMetrics:
    """Compute ComparisonMetrics from matching results."""
    num_matched = len(matched_pairs)
    num_fp = len(false_positives)
    num_missed = len(missed_detections)
    total_predictions = num_matched + num_fp
    total_gt = num_matched + num_missed

    precision = num_matched / total_predictions if total_predictions > 0 else 1.0
    recall = num_matched / total_gt if total_gt > 0 else 1.0

    if num_matched > 0:
        avg_iou = sum(compute_iou(p, g) for p, g in matched_pairs) / num_matched
        class_mismatches = sum(1 for p, g in matched_pairs if p.class_id != g.class_id)
        class_mismatch_rate = class_mismatches / num_matched
    else:
        avg_iou = 0.0
        class_mismatch_rate = 0.0

    return ComparisonMetrics(
        precision=precision,
        recall=recall,
        avg_iou=avg_iou,
        class_mismatch_rate=class_mismatch_rate,
        false_positive_count=num_fp,
        missed_count=num_missed,
    )


def compare_inference(
    baseline: list[ImageDetectionResult],
    candidate: list[ImageDetectionResult],
    iou_threshold: float = 0.5,
) -> ComparisonSummary:
    """Compare candidate detections against baseline detections.

    Matches images by image_path. Images present in one set but not the
    other contribute as fully missed or fully false-positive.
    """
    baseline_by_path = {r.image_path: r for r in baseline}
    candidate_by_path = {r.image_path: r for r in candidate}

    all_paths = sorted(set(baseline_by_path) | set(candidate_by_path))

    per_image: list[ComparisonMetrics] = []
    all_matched: list[tuple[Detection, Detection]] = []
    all_fp: list[Detection] = []
    all_missed: list[Detection] = []

    for path in all_paths:
        baseline_dets = baseline_by_path[path].detections if path in baseline_by_path else []
        candidate_dets = candidate_by_path[path].detections if path in candidate_by_path else []

        matched, fp, missed = match_detections(candidate_dets, baseline_dets, iou_threshold)
        per_image.append(_compute_metrics(matched, fp, missed))
        all_matched.extend(matched)
        all_fp.extend(fp)
        all_missed.extend(missed)

    overall = _compute_metrics(all_matched, all_fp, all_missed)

    return ComparisonSummary(per_image=per_image, overall=overall)


def compare_backends(
    results_by_backend: dict[str, list[ImageDetectionResult]],
    baseline_key: str,
) -> dict[str, ComparisonSummary]:
    """Compare multiple backends against a baseline.

    Returns a dict mapping each non-baseline backend key to its ComparisonSummary.
    """
    baseline = results_by_backend[baseline_key]
    return {
        key: compare_inference(baseline, candidate)
        for key, candidate in results_by_backend.items()
        if key != baseline_key
    }
