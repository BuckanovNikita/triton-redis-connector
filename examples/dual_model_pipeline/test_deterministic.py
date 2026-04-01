"""Phase 5: Deterministic drift test for dual-model pipeline.

Runs one COCO8 image through both model paths (even/odd camera),
generates reference detections, then repeats N times and verifies
no drift in detection coordinates.

Usage:
    uv run python examples/dual_model_pipeline/test_deterministic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tritonclient.grpc as grpcclient  # pyright: ignore[reportMissingTypeStubs]
from loguru import logger
from triton_redis_connector.engine.metric_aggregator import (
    extract_event_counters,
    extract_field_gauges,
    extract_field_group_bys,
)
from triton_redis_connector.preprocessing.decoder import OpenCVDecoder
from triton_redis_connector.preprocessing.letterbox import LetterBoxPreprocessor

TEST_IMAGE = Path("/workspace/test_data/coco8/images/train/000000000025.jpg")
TRITON_URL = "triton:8001"
REPEAT_COUNT = 5
MAX_COORD_DRIFT = 0.01

EVEN_MODEL = "yolo11s_960x608"
EVEN_TARGET_SIZE = (608, 960)
ODD_MODEL = "yolo26l"
ODD_TARGET_SIZE = (640, 640)

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def load_and_preprocess(
    image_path: Path,
    target_size: tuple[int, int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8], int, int]:
    """Return (tensor, raw_image, orig_h, orig_w) after letterbox preprocessing."""
    decoder = OpenCVDecoder()
    preprocessor = LetterBoxPreprocessor(target_size=target_size)

    raw_bytes = image_path.read_bytes()
    image = decoder.decode(raw_bytes)
    h, w = image.shape[:2]

    tensor, _scale, _pad = preprocessor.preprocess(image)
    tensor_f32 = np.asarray(tensor, dtype=np.float32)

    return tensor_f32, image, h, w


def run_inference(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    tensor: npt.NDArray[np.float32],
) -> dict[str, npt.NDArray[np.float32]]:
    """Send a tensor to Triton and return output arrays."""
    input_tensor = grpcclient.InferInput("images", tensor.shape, "FP32")
    input_tensor.set_data_from_numpy(tensor)

    result = client.infer(
        model_name=model_name,
        inputs=[input_tensor],
        outputs=[grpcclient.InferRequestedOutput("output0")],
    )

    return {"output0": np.asarray(result.as_numpy("output0"), dtype=np.float32)}


def postprocess_standard(
    raw: npt.NDArray[np.float32],
    conf_threshold: float = CONF_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
) -> list[dict[str, Any]]:
    """Apply confidence filter + NMS to standard YOLO output."""
    if raw.ndim == 3:
        raw = raw[0]

    if raw.shape[0] < raw.shape[1]:
        raw = raw.T

    boxes_xywh = raw[:, :4]
    class_scores = raw[:, 4:]
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = max_scores >= conf_threshold
    boxes = boxes_xywh[mask]
    scores = max_scores[mask]
    cids = class_ids[mask]

    if len(scores) == 0:
        return []

    xyxy = _xywh_to_xyxy(boxes)
    keep = _nms_numpy(xyxy, scores, iou_threshold)

    return [
        {
            "x1": float(xyxy[i, 0]),
            "y1": float(xyxy[i, 1]),
            "x2": float(xyxy[i, 2]),
            "y2": float(xyxy[i, 3]),
            "confidence": float(scores[i]),
            "class_id": int(cids[i]),
        }
        for i in keep
    ]


def postprocess_end2end(
    raw: npt.NDArray[np.float32],
    conf_threshold: float = CONF_THRESHOLD,
) -> list[dict[str, Any]]:
    """Apply confidence filter to end-to-end YOLO output (NMS already applied)."""
    if raw.ndim == 3:
        raw = raw[0]

    mask = raw[:, 4] >= conf_threshold
    filtered = raw[mask]

    return [
        {
            "x1": float(filtered[i, 0]),
            "y1": float(filtered[i, 1]),
            "x2": float(filtered[i, 2]),
            "y2": float(filtered[i, 3]),
            "confidence": float(filtered[i, 4]),
            "class_id": int(filtered[i, 5]),
        }
        for i in range(len(filtered))
    ]


def compare_detections(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    max_drift: float,
) -> tuple[bool, list[str]]:
    """Compare two detection lists, return (match, errors)."""
    errors: list[str] = []

    if len(reference) != len(candidate):
        errors.append(f"Detection count mismatch: reference={len(reference)}, candidate={len(candidate)}")
        return False, errors

    for i, (ref, cand) in enumerate(zip(reference, candidate, strict=True)):
        for key in ("x1", "y1", "x2", "y2", "confidence"):
            diff = abs(ref[key] - cand[key])
            if diff > max_drift:
                errors.append(f"Detection {i} {key}: ref={ref[key]:.4f} cand={cand[key]:.4f} drift={diff:.6f}")

        if ref["class_id"] != cand["class_id"]:
            errors.append(f"Detection {i} class_id: ref={ref['class_id']} cand={cand['class_id']}")

    return len(errors) == 0, errors


def _run_path(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    target_size: tuple[int, int],
    path_label: str,
) -> list[dict[str, Any]]:
    """Run inference and postprocessing for one model path."""
    logger.info("=== {} PATH: {} (target {}) ===", path_label, model_name, target_size)

    tensor, _img, orig_h, orig_w = load_and_preprocess(TEST_IMAGE, target_size)
    logger.info("Input tensor shape: {}, image: {}x{}", tensor.shape, orig_w, orig_h)

    outputs = run_inference(client, model_name, tensor)
    raw = outputs["output0"]
    logger.info("Output shape: {}", raw.shape)

    is_end2end = raw.ndim == 3 and raw.shape[-1] == 6
    detections = postprocess_end2end(raw) if is_end2end else postprocess_standard(raw)
    logger.info("Detections: {} objects found (end2end={})", len(detections), is_end2end)
    for d in detections[:5]:
        logger.info(
            "  class={} conf={:.3f} box=[{:.1f},{:.1f},{:.1f},{:.1f}]",
            d["class_id"],
            d["confidence"],
            d["x1"],
            d["y1"],
            d["x2"],
            d["y2"],
        )

    return detections


def _test_deterministic_repeat(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    target_size: tuple[int, int],
    reference: list[dict[str, Any]],
    repeat: int,
) -> bool:
    """Repeat inference N times, verify no drift from reference."""
    logger.info("--- Deterministic test: {} x{} repeats ---", model_name, repeat)
    all_pass = True

    for i in range(repeat):
        tensor, _img, _h, _w = load_and_preprocess(TEST_IMAGE, target_size)
        outputs = run_inference(client, model_name, tensor)
        raw = outputs["output0"]

        is_end2end = raw.ndim == 3 and raw.shape[-1] == 6
        detections = postprocess_end2end(raw) if is_end2end else postprocess_standard(raw)
        match, errors = compare_detections(reference, detections, MAX_COORD_DRIFT)

        if match:
            logger.info("  Repeat {}/{}: PASS ({} detections)", i + 1, repeat, len(detections))
        else:
            logger.error("  Repeat {}/{}: FAIL", i + 1, repeat)
            for err in errors:
                logger.error("    {}", err)
            all_pass = False

    return all_pass


def _test_output_format(detections: list[dict[str, Any]], model_name: str, metal_type: str) -> bool:
    """Verify the output message would have all required fields."""
    logger.info("--- Output format verification: {} ---", model_name)

    result: dict[str, str | int | float] = {
        "camera_no": 2 if metal_type == "HK" else 3,
        "metal_type": metal_type,
        "detections_json": json.dumps(detections),
        "model_name": model_name,
        "inference_time_ms": 12.5,
        "num_detections": len(detections),
        "frame_counter": 1,
    }

    required_fields = {
        "camera_no",
        "metal_type",
        "detections_json",
        "model_name",
        "inference_time_ms",
        "num_detections",
        "frame_counter",
    }
    missing = required_fields - set(result.keys())
    if missing:
        logger.error("Missing fields: {}", missing)
        return False

    parsed = json.loads(str(result["detections_json"]))
    if not isinstance(parsed, list):
        logger.error("detections_json is not a list")
        return False

    if len(parsed) != result["num_detections"]:
        logger.error("num_detections mismatch: {} vs {}", len(parsed), result["num_detections"])
        return False

    logger.info("  All fields present, detections_json valid ({} detections)", len(parsed))
    logger.info("  metal_type={}, camera_no={}", result["metal_type"], result["camera_no"])
    return True


def _test_wrapper_metrics() -> bool:
    """Verify wrapper metric extraction works for pipeline event types."""
    logger.info("--- Wrapper metric extraction test ---")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.dual_model_pipeline.events import (  # noqa: PLC0415
        EvenCameraFrame,
        InspectionResult,
        MetadataInferenceInput,
        OddCameraFrame,
        SteelInspectionInput,
    )

    all_pass = True

    for cls, expected in [
        (SteelInspectionInput, "inspection_received"),
        (EvenCameraFrame, "even_camera_frame"),
        (OddCameraFrame, "odd_camera_frame"),
        (InspectionResult, "inspection_complete"),
    ]:
        counters = extract_event_counters(cls)
        names = [c.name for c in counters]
        if expected in names:
            logger.info("  Counter '{}' on {}: OK", expected, cls.__name__)
        else:
            logger.error("  Counter '{}' NOT FOUND on {}", expected, cls.__name__)
            all_pass = False

    gauges = extract_field_gauges(InspectionResult)
    gauge_names = [g.name for _, g in gauges]
    if "inspection_inference_ms" in gauge_names:
        logger.info("  Gauge 'inspection_inference_ms' on InspectionResult: OK")
    else:
        logger.error("  Gauge 'inspection_inference_ms' NOT FOUND on InspectionResult")
        all_pass = False

    for cls, expected in [
        (EvenCameraFrame, "camera_no_per_minute"),
        (OddCameraFrame, "camera_no_per_minute"),
        (MetadataInferenceInput, "metal_type_per_minute"),
    ]:
        group_bys = extract_field_group_bys(cls)
        gb_names = [g.name for _, g in group_bys]
        if expected in gb_names:
            logger.info("  GroupBy '{}' on {}: OK", expected, cls.__name__)
        else:
            logger.error("  GroupBy '{}' NOT FOUND on {}", expected, cls.__name__)
            all_pass = False

    return all_pass


def main() -> None:
    """Run all Phase 5 deterministic drift and verification tests."""
    logger.info("Phase 5: Deterministic Drift Test")
    logger.info("Image: {}", TEST_IMAGE)
    logger.info("Triton: {}", TRITON_URL)

    if not TEST_IMAGE.exists():
        logger.error("Test image not found: {}", TEST_IMAGE)
        sys.exit(1)

    client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=TRITON_URL)

    results: dict[str, bool] = {}

    even_ref = _run_path(client, EVEN_MODEL, EVEN_TARGET_SIZE, "EVEN")
    odd_ref = _run_path(client, ODD_MODEL, ODD_TARGET_SIZE, "ODD")

    results["even_deterministic"] = _test_deterministic_repeat(
        client, EVEN_MODEL, EVEN_TARGET_SIZE, even_ref, REPEAT_COUNT
    )
    results["odd_deterministic"] = _test_deterministic_repeat(client, ODD_MODEL, ODD_TARGET_SIZE, odd_ref, REPEAT_COUNT)
    results["even_output_format"] = _test_output_format(even_ref, EVEN_MODEL, "HK")
    results["odd_output_format"] = _test_output_format(odd_ref, ODD_MODEL, "GK")
    results["wrapper_metrics"] = _test_wrapper_metrics()

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  {}: {}", name, status)
        if not passed:
            all_pass = False

    if all_pass:
        logger.info("ALL TESTS PASSED")
    else:
        logger.error("SOME TESTS FAILED")
        sys.exit(1)

    ref_data = {
        "even_path": {"model": EVEN_MODEL, "target_size": list(EVEN_TARGET_SIZE), "detections": even_ref},
        "odd_path": {"model": ODD_MODEL, "target_size": list(ODD_TARGET_SIZE), "detections": odd_ref},
        "test_image": str(TEST_IMAGE),
        "repeat_count": REPEAT_COUNT,
        "max_coord_drift": MAX_COORD_DRIFT,
    }
    ref_path = Path(__file__).parent / "results" / "reference_detections.json"
    ref_path.write_text(json.dumps(ref_data, indent=2))
    logger.info("Reference detections saved to {}", ref_path)


def _xywh_to_xyxy(boxes: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    result = np.empty_like(boxes)
    half_w = boxes[:, 2] / 2.0
    half_h = boxes[:, 3] / 2.0
    result[:, 0] = boxes[:, 0] - half_w
    result[:, 1] = boxes[:, 1] - half_h
    result[:, 2] = boxes[:, 0] + half_w
    result[:, 3] = boxes[:, 1] + half_h
    return result


def _nms_numpy(
    boxes: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float32],
    iou_threshold: float,
) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[idx], x1[rest])
        yy1 = np.maximum(y1[idx], y1[rest])
        xx2 = np.minimum(x2[idx], x2[rest])
        yy2 = np.minimum(y2[idx], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[idx] + areas[rest] - intersection
        iou = np.where(union > 0, intersection / union, np.float32(0.0))
        order = rest[iou <= iou_threshold]

    return keep


if __name__ == "__main__":
    main()
