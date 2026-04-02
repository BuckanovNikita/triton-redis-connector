"""Pipeline drift test: our preprocessing + ONNX inference + postprocessing vs ultralytics.

Task 3.11b — verifies that our full pipeline produces results matching ultralytics
reference predictions. Any significant drift indicates a bug in our preprocessing
or postprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from triton_model_tools.benchmarks.comparison import compare_inference
from triton_model_tools.inference.postprocessing import YoloPostprocessor
from triton_model_tools.models.detection import Detection, ImageDetectionResult
from triton_shared_models.preprocessing import LetterBoxPreprocessor as YoloPreprocessor

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "test_fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"
MODELS_DIR = FIXTURES_DIR / "models"

YOLO_MODEL_NAME = "yolo11s"
YOLO_PT_NAME = "yolo11s.pt"
TARGET_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

MIN_PRECISION = 0.95
MIN_RECALL = 0.95
COMPARISON_IOU_THRESHOLD = 0.5


def _get_onnx_path() -> Path | None:
    """Find the ONNX model in fixtures or common locations."""
    candidates = [
        MODELS_DIR / "yolo11s.onnx",
        MODELS_DIR / "yolo11s" / "yolo11s.onnx",
        Path("yolo11s.onnx"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _ensure_ultralytics_model() -> Any:
    """Load ultralytics YOLO model, skip if unavailable."""
    try:
        from ultralytics import YOLO
    except ImportError:
        pytest.skip("ultralytics not installed")

    try:
        model: Any = YOLO(YOLO_PT_NAME)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot load ultralytics model {YOLO_PT_NAME}: {exc}")
    return model


def _ensure_onnx_model() -> Path:
    """Find or export ONNX model, skip if not possible."""
    existing = _get_onnx_path()
    if existing is not None:
        return existing

    try:
        from ultralytics import YOLO
    except ImportError:
        pytest.skip("No ONNX model and ultralytics not installed for export")

    try:
        model: Any = YOLO(YOLO_PT_NAME)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        export_path: str = model.export(format="onnx", imgsz=TARGET_SIZE, dynamic=False)
        target = MODELS_DIR / "yolo11s.onnx"
        if Path(export_path).resolve() != target.resolve():
            Path(export_path).rename(target)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot export ONNX model: {exc}")
    else:
        return target


def _get_test_images() -> list[Path]:
    """Collect test fixture images."""
    if not IMAGES_DIR.is_dir():
        pytest.skip(f"Test images directory not found: {IMAGES_DIR}")
    paths = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        pytest.skip("No test images found")
    return paths


def _ultralytics_results_to_detections(
    results: Any,
    image_path: str,
) -> ImageDetectionResult:
    """Convert ultralytics prediction results to our ImageDetectionResult format."""
    detections: list[Detection] = []
    for result in results:
        boxes: Any = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy: Any = boxes.xyxy.cpu().numpy()
        confs: Any = boxes.conf.cpu().numpy()
        cls_ids: Any = boxes.cls.cpu().numpy().astype(int)
        detections.extend(
            Detection(
                x1=float(xyxy[i, 0]),
                y1=float(xyxy[i, 1]),
                x2=float(xyxy[i, 2]),
                y2=float(xyxy[i, 3]),
                confidence=float(confs[i]),
                class_id=int(cls_ids[i]),
            )
            for i in range(len(xyxy))
        )
    return ImageDetectionResult(
        image_path=image_path,
        model_name=YOLO_MODEL_NAME,
        detections=detections,
        inference_time_ms=0.0,
    )


def _run_our_pipeline(
    image_paths: list[Path],
    onnx_path: Path,
) -> list[ImageDetectionResult]:
    """Run images through our pipeline: LetterBoxPreprocessor -> ONNX -> YoloPostprocessor."""
    from triton_model_tools.inference.onnx_inference import OnnxInferenceClient

    client = OnnxInferenceClient(onnx_path)
    preprocessor = YoloPreprocessor(target_size=TARGET_SIZE)
    end2end = client.detect_end2end()
    postprocessor = YoloPostprocessor(end2end=end2end)

    results: list[ImageDetectionResult] = []
    for image_path in image_paths:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            continue

        tensor, scale, pad_offset = preprocessor.preprocess(bgr)
        tensor_f32 = tensor.astype(np.float32)
        outputs = client.infer(tensor_f32)
        detections = postprocessor.postprocess(
            outputs,
            scale,
            pad_offset,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
        )
        results.append(
            ImageDetectionResult(
                image_path=image_path.name,
                model_name=YOLO_MODEL_NAME,
                detections=detections,
                inference_time_ms=0.0,
            ),
        )
    return results


def _run_ultralytics_reference(
    model: Any,
    image_paths: list[Path],
) -> list[ImageDetectionResult]:
    """Run ultralytics predict on all images."""
    results: list[ImageDetectionResult] = []
    for image_path in image_paths:
        preds: Any = model.predict(
            str(image_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=TARGET_SIZE,
            verbose=False,
        )
        det_result = _ultralytics_results_to_detections(preds, image_path.name)
        results.append(det_result)
    return results


@pytest.mark.slow
class TestPipelineDriftVsUltralytics:
    """Task 3.11b: Our pipeline must closely match ultralytics reference detections."""

    def test_detection_precision_matches_ultralytics(self) -> None:
        image_paths = _get_test_images()
        onnx_path = _ensure_onnx_model()
        ul_model = _ensure_ultralytics_model()

        reference = _run_ultralytics_reference(ul_model, image_paths)
        ours = _run_our_pipeline(image_paths, onnx_path)

        summary = compare_inference(
            reference,
            ours,
            iou_threshold=COMPARISON_IOU_THRESHOLD,
        )

        assert summary.overall.precision >= MIN_PRECISION, (
            f"Precision {summary.overall.precision:.4f} < {MIN_PRECISION}. "
            f"FP={summary.overall.false_positive_count}, missed={summary.overall.missed_count}"
        )

    def test_detection_recall_matches_ultralytics(self) -> None:
        image_paths = _get_test_images()
        onnx_path = _ensure_onnx_model()
        ul_model = _ensure_ultralytics_model()

        reference = _run_ultralytics_reference(ul_model, image_paths)
        ours = _run_our_pipeline(image_paths, onnx_path)

        summary = compare_inference(
            reference,
            ours,
            iou_threshold=COMPARISON_IOU_THRESHOLD,
        )

        assert summary.overall.recall >= MIN_RECALL, (
            f"Recall {summary.overall.recall:.4f} < {MIN_RECALL}. missed={summary.overall.missed_count}"
        )

    def test_detection_count_is_close(self) -> None:
        image_paths = _get_test_images()
        onnx_path = _ensure_onnx_model()
        ul_model = _ensure_ultralytics_model()

        reference = _run_ultralytics_reference(ul_model, image_paths)
        ours = _run_our_pipeline(image_paths, onnx_path)

        ref_total = sum(len(r.detections) for r in reference)
        our_total = sum(len(r.detections) for r in ours)

        if ref_total == 0 and our_total == 0:
            return

        if ref_total > 0:
            ratio = our_total / ref_total
            assert 0.8 <= ratio <= 1.2, (
                f"Detection count ratio {ratio:.2f} outside [0.8, 1.2]. Reference={ref_total}, ours={our_total}"
            )

    def test_class_mismatch_rate_is_low(self) -> None:
        image_paths = _get_test_images()
        onnx_path = _ensure_onnx_model()
        ul_model = _ensure_ultralytics_model()

        reference = _run_ultralytics_reference(ul_model, image_paths)
        ours = _run_our_pipeline(image_paths, onnx_path)

        summary = compare_inference(
            reference,
            ours,
            iou_threshold=COMPARISON_IOU_THRESHOLD,
        )

        assert summary.overall.class_mismatch_rate <= 0.05, (
            f"Class mismatch rate {summary.overall.class_mismatch_rate:.4f} > 0.05"
        )
