"""Prediction comparison utilities for evaluating detection and classification model outputs.

Provides detection matching with greedy algorithm,
multi-backend comparison summaries, and classification accuracy metrics.
"""

import torch
import torchvision.ops
from pydantic import BaseModel, ConfigDict, Field

from triton_model_tools.models.classification import ImageClassificationResult
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


def _detections_to_tensor(detections: list[Detection]) -> torch.Tensor:
    return torch.tensor(
        [[d.x1, d.y1, d.x2, d.y2] for d in detections],
        dtype=torch.float32,
    )


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

    pred_boxes = _detections_to_tensor(predictions)
    gt_boxes = _detections_to_tensor(ground_truth)
    iou_matrix_tensor = torchvision.ops.box_iou(pred_boxes, gt_boxes)

    candidates: list[tuple[float, int, int]] = []
    for p_idx in range(len(predictions)):
        for g_idx in range(len(ground_truth)):
            iou_val = float(iou_matrix_tensor[p_idx, g_idx])
            if iou_val >= iou_threshold:
                candidates.append((iou_val, p_idx, g_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_pred_indices: set[int] = set()
    matched_gt_indices: set[int] = set()
    matched_pairs: list[tuple[Detection, Detection]] = []

    for _iou_val, p_idx, g_idx in candidates:
        if p_idx in matched_pred_indices or g_idx in matched_gt_indices:
            continue
        matched_pairs.append((predictions[p_idx], ground_truth[g_idx]))
        matched_pred_indices.add(p_idx)
        matched_gt_indices.add(g_idx)

    false_positives = [p for i, p in enumerate(predictions) if i not in matched_pred_indices]
    missed_detections = [g for i, g in enumerate(ground_truth) if i not in matched_gt_indices]

    return matched_pairs, false_positives, missed_detections


def _compute_iou_for_pair(pred: Detection, gt: Detection) -> float:
    pred_t = torch.tensor([[pred.x1, pred.y1, pred.x2, pred.y2]], dtype=torch.float32)
    gt_t = torch.tensor([[gt.x1, gt.y1, gt.x2, gt.y2]], dtype=torch.float32)
    return float(torchvision.ops.box_iou(pred_t, gt_t)[0, 0])


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
        avg_iou = sum(_compute_iou_for_pair(p, g) for p, g in matched_pairs) / num_matched
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


# ---------------------------------------------------------------------------
# Classification comparison
# ---------------------------------------------------------------------------

TOP_5_K = 5


class ClassificationComparisonMetrics(BaseModel):
    """Accuracy metrics for comparing classification results."""

    model_config = ConfigDict(frozen=True)

    top1_accuracy: float = Field(ge=0.0, le=1.0)
    top5_accuracy: float = Field(ge=0.0, le=1.0)
    class_mismatch_rate: float = Field(ge=0.0, le=1.0)
    total_images: int = Field(ge=0)


class ClassificationComparisonSummary(BaseModel):
    """Per-image and overall classification comparison."""

    model_config = ConfigDict(frozen=True)

    per_image: list[ClassificationComparisonMetrics]
    overall: ClassificationComparisonMetrics


def _compare_single_image(
    baseline_result: ImageClassificationResult,
    candidate_result: ImageClassificationResult,
) -> ClassificationComparisonMetrics:
    """Compare classification results for a single image pair."""
    baseline_top1_id = baseline_result.predictions[0].class_id if baseline_result.predictions else -1
    candidate_top1_id = candidate_result.predictions[0].class_id if candidate_result.predictions else -2

    top1_match = int(baseline_top1_id == candidate_top1_id)

    candidate_top5_ids = {p.class_id for p in candidate_result.predictions[:TOP_5_K]}
    top5_match = int(baseline_top1_id in candidate_top5_ids) if baseline_result.predictions else 0

    class_mismatch = 1.0 - float(top1_match)

    return ClassificationComparisonMetrics(
        top1_accuracy=float(top1_match),
        top5_accuracy=float(top5_match),
        class_mismatch_rate=class_mismatch,
        total_images=1,
    )


def compare_classification(
    baseline: list[ImageClassificationResult],
    candidate: list[ImageClassificationResult],
) -> ClassificationComparisonSummary:
    """Compare candidate classification results against baseline.

    Matches images by image_path. For each matched pair:
    - Checks if top-1 prediction class_id matches between baseline and candidate
    - Checks if baseline top-1 class_id appears in candidate top-5
    - Computes class mismatch rate (1 - top1_accuracy)

    Images present in only one set are treated as mismatches.
    """
    baseline_by_path = {r.image_path: r for r in baseline}
    candidate_by_path = {r.image_path: r for r in candidate}

    all_paths = sorted(set(baseline_by_path) | set(candidate_by_path))

    per_image: list[ClassificationComparisonMetrics] = []
    total_top1_matches = 0
    total_top5_matches = 0

    for path in all_paths:
        if path in baseline_by_path and path in candidate_by_path:
            metrics = _compare_single_image(baseline_by_path[path], candidate_by_path[path])
            per_image.append(metrics)
            total_top1_matches += int(metrics.top1_accuracy > 0)
            total_top5_matches += int(metrics.top5_accuracy > 0)
        else:
            per_image.append(
                ClassificationComparisonMetrics(
                    top1_accuracy=0.0,
                    top5_accuracy=0.0,
                    class_mismatch_rate=1.0,
                    total_images=1,
                ),
            )

    total = len(all_paths)
    overall_top1 = total_top1_matches / total if total > 0 else 1.0
    overall_top5 = total_top5_matches / total if total > 0 else 1.0

    overall = ClassificationComparisonMetrics(
        top1_accuracy=overall_top1,
        top5_accuracy=overall_top5,
        class_mismatch_rate=1.0 - overall_top1,
        total_images=total,
    )

    return ClassificationComparisonSummary(per_image=per_image, overall=overall)
