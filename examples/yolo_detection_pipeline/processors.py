"""Pipeline-specific processors for the YOLO detection pipeline.

These processors wire the framework building blocks (OpenCVDecoder,
LetterBoxPreprocessor, InferenceWorker) into a complete pipeline graph.
They live in examples/ because they cross import-linter sibling boundaries
(preprocessing + inference types in the same file).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from loguru import logger
from triton_redis_connector.engine.events import DecodedImage, DetectionResult, InferenceInput
from triton_redis_connector.preprocessing.decoder import OpenCVDecoder
from triton_redis_connector.preprocessing.letterbox import LetterBoxPreprocessor
from triton_redis_connector.protocols import Envelope

if TYPE_CHECKING:
    from triton_redis_connector.errors.models import InferenceError
    from triton_redis_connector.inference.request import InferenceOutput

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45


class EnvelopeExtractor:
    """Extracts raw bytes payload from Envelope[bytes].

    Bridges RedisStreamSource (which produces Envelope[bytes]) to
    downstream processors that consume raw bytes.
    """

    def __call__(self, event: Envelope[bytes]) -> bytes | None:
        """Extract payload bytes from an Envelope."""
        if event.payload:
            return event.payload
        return None


class ImageDeserializer:
    """Decodes raw image bytes into DecodedImage.

    Reads raw bytes from the pipeline, decodes using OpenCV, and produces
    DecodedImage events for downstream preprocessing.
    """

    def __init__(self) -> None:  # noqa: D107
        self._decoder = OpenCVDecoder()

    def __call__(self, event: bytes) -> DecodedImage | None:
        """Decode raw image bytes into a DecodedImage."""
        try:
            image = self._decoder.decode(event)
            h, w = image.shape[:2]
            return DecodedImage(
                image=image,
                original_height=h,
                original_width=w,
            )
        except ValueError:
            logger.exception("Failed to decode image")
            return None


class PreprocessorNode:
    """Letterbox preprocessing + InferenceInput construction.

    Transforms a decoded BGR image into a model-ready float32 tensor
    using the letterbox algorithm, then wraps it as InferenceInput.
    """

    def __init__(  # noqa: D107
        self,
        model_name: str = "yolo11s",
        target_size: tuple[int, int] = (640, 640),
    ) -> None:
        self._preprocessor = LetterBoxPreprocessor(target_size=target_size)
        self._model_name = model_name

    def __call__(self, event: DecodedImage) -> InferenceInput | None:
        """Preprocess decoded image into an InferenceInput tensor."""
        tensor, _scale, _pad = self._preprocessor.preprocess(event.image)
        return InferenceInput(
            model_name=self._model_name,
            raw_image=tensor,
            input_name="images",
            output_names=("output0",),
        )


class DetectionPostProcessor:
    """Parses YOLO output tensors into DetectionResult.

    Implements standard YOLO postprocessing: confidence filtering,
    xywh-to-xyxy conversion, and NMS. Ported from triton_model_tools
    (cannot import directly due to import-linter package boundary).
    """

    def __init__(  # noqa: D107
        self,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold

    def __call__(self, event: InferenceOutput) -> DetectionResult | None:
        """Parse YOLO output0 tensor into detection results JSON."""
        raw = event.outputs.get("output0")
        if raw is None:
            logger.warning("InferenceOutput missing 'output0' tensor for model {}", event.model_name)
            return None

        detections = self._postprocess_standard(raw)
        detections_json = json.dumps(detections)

        return DetectionResult(
            detections_json=detections_json,
            model_name=event.model_name,
            inference_time_ms=event.inference_time_ms,
            num_detections=len(detections),
        )

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
    """Serializes DetectionResult to Envelope[bytes] for Redis sink.

    Converts DetectionResult pydantic model to JSON bytes wrapped in an
    Envelope for the RedisStreamSink to write via XADD.
    """

    def __call__(self, event: DetectionResult) -> Envelope[bytes] | None:
        """Serialize DetectionResult to Envelope[bytes]."""
        payload = event.model_dump_json().encode()
        return Envelope(
            payload=payload,
            source_stream=event.source_stream,
            message_id=event.message_id,
        )


class ErrorLogSink:
    """Logs inference errors without dropping them silently."""

    def __call__(self, event: InferenceError) -> None:
        """Log the inference error."""
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
