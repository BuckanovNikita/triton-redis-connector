"""Pipeline processors for the dual-model steel inspection pipeline.

These processors wire framework building blocks into a branching pipeline graph.
They live in examples/ because they cross import-linter sibling boundaries
(preprocessing + inference types in the same file).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger
from triton_redis_connector.engine.events import InferenceInput
from triton_redis_connector.errors.models import InferenceError
from triton_redis_connector.inference.worker import InferenceWorker
from triton_redis_connector.preprocessing.decoder import TurboJPEGDecoder
from triton_redis_connector.preprocessing.letterbox import LetterBoxPreprocessor
from triton_redis_connector.protocols import Envelope

from examples.dual_model_pipeline.events import (
    EvenCameraFrame,
    InspectionResult,
    MetadataInferenceInput,
    MetadataInferenceOutput,
    OddCameraFrame,
    SteelInspectionInput,
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

DALI_ENSEMBLE_MODEL = "yolo11s_960x608_dali_ensemble"
YOLO26L_MODEL = "yolo26l"


class InputExtractor:
    """Extracts SteelInspectionInput from a Redis stream Envelope[bytes]."""

    def __call__(self, event: Envelope[bytes]) -> SteelInspectionInput | None:
        """Parse original_fields from the envelope into a typed inspection input."""
        fields = event.original_fields
        camera_no_raw = fields.get("camera_no")
        image_path = fields.get("image_path")
        frame_counter_raw = fields.get("frame_counter")
        receipt_ts_raw = fields.get("receipt_ts")

        if not all((camera_no_raw, image_path, frame_counter_raw)):
            logger.warning(
                "Missing required fields in envelope {}: camera_no={}, image_path={}, frame_counter={}",
                event.message_id,
                camera_no_raw,
                image_path,
                frame_counter_raw,
            )
            return None

        return SteelInspectionInput(
            receipt_ts=float(receipt_ts_raw) if receipt_ts_raw else time.monotonic(),
            camera_no=int(camera_no_raw),  # type: ignore[arg-type]
            image_path=str(image_path),
            frame_counter=int(frame_counter_raw),  # type: ignore[arg-type]
            metal_type=fields.get("metal_type"),
        )


class CameraBranchRouter:
    """Routes inspection input to even/odd camera frame events.

    Even camera_no -> EvenCameraFrame (metal_type="HK", DALI path)
    Odd camera_no  -> OddCameraFrame  (metal_type="GK", TurboJPEG path)
    """

    def __call__(self, event: SteelInspectionInput) -> EvenCameraFrame | OddCameraFrame | None:
        """Read image from filesystem and route based on camera_no parity."""
        image_path = Path(event.image_path)
        try:
            image_bytes = image_path.read_bytes()
        except (FileNotFoundError, PermissionError, OSError):
            logger.exception("Failed to read image at {}", event.image_path)
            return None

        if not image_bytes:
            logger.warning("Empty image file at {}", event.image_path)
            return None

        if event.camera_no % 2 == 0:
            return EvenCameraFrame(
                image_bytes=image_bytes,
                camera_no=event.camera_no,
                frame_counter=event.frame_counter,
                metal_type="HK",
            )
        return OddCameraFrame(
            image_bytes=image_bytes,
            camera_no=event.camera_no,
            frame_counter=event.frame_counter,
            metal_type="GK",
        )


class DaliPreprocessor:
    """Wraps raw image bytes into an InferenceInput for the DALI ensemble.

    The DALI ensemble handles decoding and preprocessing on the Triton side,
    so we pass raw JPEG/PNG bytes as UINT8 input.
    """

    def __call__(self, event: EvenCameraFrame) -> MetadataInferenceInput | None:
        """Wrap raw bytes as InferenceInput for DALI ensemble."""
        inference_input = InferenceInput(
            model_name=DALI_ENSEMBLE_MODEL,
            image_bytes=event.image_bytes,
            input_name="INPUT_IMAGE",
            output_names=("OUTPUT_DETECTIONS",),
        )
        return MetadataInferenceInput(
            inference_input=inference_input,
            camera_no=event.camera_no,
            frame_counter=event.frame_counter,
            metal_type=event.metal_type,
        )


class TurboJpegPreprocessor:
    """Decodes image via TurboJPEG and preprocesses with LetterBox for yolo26l.

    Standard client-side preprocessing path:
      raw bytes -> TurboJPEG decode -> BGR image -> LetterBox -> FP32 tensor
    """

    def __init__(self, target_size: tuple[int, int] = (640, 640)) -> None:  # noqa: D107
        self._decoder = TurboJPEGDecoder()
        self._preprocessor = LetterBoxPreprocessor(target_size=target_size)

    def __call__(self, event: OddCameraFrame) -> MetadataInferenceInput | None:
        """Decode and preprocess odd camera frame for yolo26l inference."""
        try:
            image = self._decoder.decode(event.image_bytes)
        except ValueError:
            logger.exception("TurboJPEG decode failed for camera {} frame {}", event.camera_no, event.frame_counter)
            return None

        tensor, _scale, _pad = self._preprocessor.preprocess(image)
        tensor_f32 = np.asarray(tensor, dtype=np.float32)

        inference_input = InferenceInput(
            model_name=YOLO26L_MODEL,
            raw_image=tensor_f32,
            input_name="images",
            output_names=("output0",),
        )
        return MetadataInferenceInput(
            inference_input=inference_input,
            camera_no=event.camera_no,
            frame_counter=event.frame_counter,
            metal_type=event.metal_type,
        )


class MetadataInferenceWorker:
    """Wraps InferenceWorker to preserve pipeline metadata across inference.

    Delegates the actual Triton gRPC call to InferenceWorker, then re-attaches
    camera_no, frame_counter, and metal_type to the output.
    """

    def __init__(self, grpc_url: str) -> None:  # noqa: D107
        self._worker = InferenceWorker(grpc_url=grpc_url)

    def __call__(self, event: MetadataInferenceInput) -> MetadataInferenceOutput | InferenceError | None:
        """Run inference and re-attach metadata on success."""
        result = self._worker(event.inference_input)

        if result is None:
            return None

        if isinstance(result, InferenceError):
            return result

        return MetadataInferenceOutput(
            inference_output=result,
            camera_no=event.camera_no,
            frame_counter=event.frame_counter,
            metal_type=event.metal_type,
        )


class InspectionPostProcessor:
    """Parses YOLO output tensors into InspectionResult.

    Handles both DALI ensemble (OUTPUT_DETECTIONS) and standard model (output0)
    output tensor names. Applies confidence filtering and NMS.
    """

    def __init__(
        self,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        """Initialize with confidence and IoU thresholds for NMS filtering."""
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold

    def __call__(self, event: MetadataInferenceOutput) -> InspectionResult | None:
        """Parse YOLO output tensor into detection results JSON."""
        output = event.inference_output
        raw: npt.NDArray[np.float32] | None = None
        for tensor_name in ("OUTPUT_DETECTIONS", "output0"):
            raw = output.outputs.get(tensor_name)
            if raw is not None:
                break

        if raw is None:
            logger.warning(
                "InferenceOutput missing expected output tensor for model {}",
                output.model_name,
            )
            return None

        is_end2end = raw.ndim == 3 and raw.shape[-1] == 6
        detections = self._postprocess_end2end(raw) if is_end2end else self._postprocess_standard(raw)
        detections_json = json.dumps(detections)

        return InspectionResult(
            camera_no=event.camera_no,
            metal_type=event.metal_type,
            detections_json=detections_json,
            model_name=output.model_name,
            inference_time_ms=output.inference_time_ms,
            num_detections=len(detections),
            frame_counter=event.frame_counter,
        )

    def _postprocess_end2end(
        self,
        raw: npt.NDArray[np.float32],
    ) -> list[dict[str, Any]]:
        if raw.ndim == 3:
            raw = raw[0]

        mask = raw[:, 4] >= self._conf_threshold
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

    def _postprocess_standard(
        self,
        raw: npt.NDArray[np.float32],
    ) -> list[dict[str, Any]]:
        if raw.ndim == 3:
            raw = raw[0]

        if raw.shape[0] < raw.shape[1]:
            raw = raw.T

        boxes_xywh = raw[:, :4]
        class_scores = raw[:, 4:]

        max_class_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        mask = max_class_scores >= self._conf_threshold
        filtered_boxes_xywh = boxes_xywh[mask]
        filtered_scores = max_class_scores[mask]
        filtered_class_ids = class_ids[mask]

        if len(filtered_scores) == 0:
            return []

        boxes_xyxy = _xywh_to_xyxy(filtered_boxes_xywh)
        keep = _nms_numpy(boxes_xyxy, filtered_scores, self._iou_threshold)

        return [
            {
                "x1": float(boxes_xyxy[idx, 0]),
                "y1": float(boxes_xyxy[idx, 1]),
                "x2": float(boxes_xyxy[idx, 2]),
                "y2": float(boxes_xyxy[idx, 3]),
                "confidence": float(filtered_scores[idx]),
                "class_id": int(filtered_class_ids[idx]),
            }
            for idx in keep
        ]


class OutputSerializer:
    """Serializes InspectionResult to Envelope[bytes] for the Redis sink."""

    def __call__(self, event: InspectionResult) -> Envelope[bytes] | None:
        """Serialize InspectionResult to JSON bytes wrapped in an Envelope."""
        payload = event.model_dump_json().encode()
        return Envelope(payload=payload)


class ErrorLogSink:
    """Logs inference errors."""

    def __call__(self, event: InferenceError) -> None:
        """Log the inference error details."""
        logger.error(
            "Inference error on model {}: {} (retries={})",
            event.model_name,
            event.error_message,
            event.retry_count,
        )


def _xywh_to_xyxy(boxes: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Convert bounding boxes from center-xywh to corner-xyxy format."""
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
    """Pure-numpy NMS implementation."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        intersection = inter_w * inter_h

        union = areas[idx] + areas[rest] - intersection
        iou = np.where(union > 0, intersection / union, np.float32(0.0))

        remaining_mask = iou <= iou_threshold
        order = rest[remaining_mask]

    return keep
