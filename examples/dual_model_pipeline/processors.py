"""Pipeline processors for the dual-model steel inspection pipeline.

These processors wire framework building blocks into a branching pipeline graph.
They live in examples/ because they cross import-linter sibling boundaries
(preprocessing + inference types in the same file).

Shipped processors used (imported in run_pipeline.py):
  - EnvelopeExtractor[SteelInspectionInput] replaces InputExtractor
  - EventSerializer[InspectionResult] replaces OutputSerializer
  - LogErrorSink replaces ErrorLogSink

Custom processors remain here because they carry domain-specific metadata
(camera_no, metal_type, frame_counter) through the inference boundary,
which the generic shipped processors do not support.
"""

from __future__ import annotations

import json
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
from triton_shared_models.nms import nms

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
                receipt_ts=event.receipt_ts,
                image_path=event.image_path,
            )
        return OddCameraFrame(
            image_bytes=image_bytes,
            camera_no=event.camera_no,
            frame_counter=event.frame_counter,
            metal_type="GK",
            receipt_ts=event.receipt_ts,
            image_path=event.image_path,
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
            receipt_ts=event.receipt_ts,
            image_path=event.image_path,
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
            receipt_ts=event.receipt_ts,
            image_path=event.image_path,
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
            receipt_ts=event.receipt_ts,
            image_path=event.image_path,
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
            receipt_ts=event.receipt_ts,
            image_path=event.image_path,
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
        keep = nms(boxes_xyxy, filtered_scores, self._iou_threshold)

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
