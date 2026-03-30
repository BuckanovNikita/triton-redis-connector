"""Triton inference client with YOLO pre/postprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from triton_model_tools.detection import Detection

if TYPE_CHECKING:
    import numpy.typing as npt

LETTERBOX_PAD_VALUE = 114
YOLO_NORMALIZE_DIVISOR = 255.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45


class TritonInferenceClient:
    """Synchronous gRPC client for Triton Inference Server."""

    def __init__(self, url: str = "localhost:8001") -> None:
        """Create a Triton gRPC inference client.

        Args:
            url: Triton server gRPC endpoint.

        """
        import tritonclient.grpc as grpcclient  # pyright: ignore[reportMissingTypeStubs]  # noqa: PLC0415

        self._client: Any = grpcclient.InferenceServerClient(url=url)
        self._grpc: Any = grpcclient

    def infer_raw(
        self,
        model_name: str,
        tensor: npt.NDArray[np.float32],
        output_names: list[str],
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Run inference with a preprocessed float32 tensor.

        Args:
            model_name: Triton model name.
            tensor: Input tensor with batch dimension (e.g. [1, 3, 640, 640]).
            output_names: Names of requested output tensors.

        Returns:
            Dict mapping output name to numpy array.

        """
        grpc = self._grpc
        input_tensor = grpc.InferInput("images", tensor.shape, "FP32")
        input_tensor.set_data_from_numpy(tensor)

        outputs = [grpc.InferRequestedOutput(name) for name in output_names]

        result = self._client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=outputs,
        )

        return {name: np.array(result.as_numpy(name), dtype=np.float32) for name in output_names}

    def infer_dali(
        self,
        model_name: str,
        image_bytes: bytes,
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Run DALI ensemble inference with raw image bytes.

        Args:
            model_name: Triton DALI ensemble model name.
            image_bytes: Raw image file bytes (JPEG/PNG).

        Returns:
            Dict mapping output name to numpy array.

        """
        grpc = self._grpc
        raw_input = np.frombuffer(image_bytes, dtype=np.uint8)
        input_array = raw_input.reshape(1, -1)

        input_tensor = grpc.InferInput("INPUT", input_array.shape, "UINT8")
        input_tensor.set_data_from_numpy(input_array)

        output = grpc.InferRequestedOutput("OUTPUT")
        result = self._client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=[output],
        )
        output_data: npt.NDArray[np.float32] = np.array(
            result.as_numpy("OUTPUT"),
            dtype=np.float32,
        )
        return {"OUTPUT": output_data}


class YoloPreprocessor:
    """Replicates ultralytics LetterBox preprocessing.

    Steps:
    1. Resize maintaining aspect ratio (scale to fit, not stretch)
    2. Pad with (114, 114, 114) to target_size (letterbox)
    3. BGR -> RGB
    4. HWC -> CHW
    5. /255.0
    6. float32
    7. batch dim

    """

    def __init__(self, target_size: int = 640) -> None:
        """Initialize with the model's expected input size."""
        self.target_size = target_size

    def preprocess(
        self,
        image: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float32], float, tuple[int, int]]:
        """Preprocess a BGR image using letterbox method.

        Args:
            image: BGR image as HWC uint8 array.

        Returns:
            Tuple of (tensor, scale_factor, (pad_left, pad_top)).

        """
        import cv2  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]

        orig_h, orig_w = image.shape[:2]
        scale = min(self.target_size / orig_h, self.target_size / orig_w)

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized: npt.NDArray[np.uint8] = cv2.resize(  # type: ignore[assignment]  # pyright: ignore[reportUnknownMemberType,reportAssignmentType]
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,  # pyright: ignore[reportUnknownMemberType]
        )

        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        padded: npt.NDArray[np.uint8] = cv2.copyMakeBorder(  # type: ignore[assignment]  # pyright: ignore[reportUnknownMemberType,reportAssignmentType]
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,  # pyright: ignore[reportUnknownMemberType]
            value=(LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE),
        )

        rgb = padded[:, :, ::-1]
        chw = np.transpose(rgb, (2, 0, 1))
        normalized = chw.astype(np.float32) / YOLO_NORMALIZE_DIVISOR
        batched: npt.NDArray[np.float32] = np.expand_dims(normalized, axis=0)

        return batched, scale, (pad_left, pad_top)


def _nms_numpy(
    boxes: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float32],
    iou_threshold: float,
) -> list[int]:
    """Pure-numpy NMS implementation.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2].
        scores: (N,) array of confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        List of kept indices.

    """
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


class YoloPostprocessor:
    """Postprocesses YOLO model outputs into Detection objects.

    Handles two output formats:
    - end2end: Already NMS'd output shaped (batch, num_dets, 6) where
      each row is [x1, y1, x2, y2, confidence, class_id].
    - standard: Raw model output shaped (batch, num_features, num_predictions)
      where features = 4 (bbox) + num_classes. Needs transpose and NMS.

    """

    def __init__(self, *, end2end: bool = False) -> None:
        """Initialize with output format flag.

        Args:
            end2end: Whether model uses end2end NMS output format.

        """
        self.end2end = end2end

    def postprocess(
        self,
        outputs: dict[str, npt.NDArray[np.float32]],
        scale: float,
        pad_offset: tuple[int, int],
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> list[Detection]:
        """Convert raw model outputs to Detection objects.

        Args:
            outputs: Dict of output name -> numpy array from Triton.
            scale: Scale factor used during preprocessing.
            pad_offset: (pad_left, pad_top) from preprocessing.
            conf_threshold: Minimum confidence to keep a detection.
            iou_threshold: IoU threshold for NMS (standard models only).

        Returns:
            List of Detection objects in original image coordinates.

        """
        if self.end2end:
            return self._postprocess_end2end(outputs, scale, pad_offset, conf_threshold)
        return self._postprocess_standard(outputs, scale, pad_offset, conf_threshold, iou_threshold)

    def _postprocess_end2end(
        self,
        outputs: dict[str, npt.NDArray[np.float32]],
        scale: float,
        pad_offset: tuple[int, int],
        conf_threshold: float,
    ) -> list[Detection]:
        output_key = next(iter(outputs))
        raw = outputs[output_key]

        if raw.ndim == 3:
            raw = raw[0]

        pad_left, pad_top = pad_offset
        detections: list[Detection] = []
        for row in raw:
            confidence = float(row[4])
            if confidence < conf_threshold:
                continue

            detections.append(
                Detection(
                    x1=(float(row[0]) - pad_left) / scale,
                    y1=(float(row[1]) - pad_top) / scale,
                    x2=(float(row[2]) - pad_left) / scale,
                    y2=(float(row[3]) - pad_top) / scale,
                    confidence=confidence,
                    class_id=int(row[5]),
                ),
            )

        logger.debug("End2end postprocess: {} detections (conf >= {})", len(detections), conf_threshold)
        return detections

    def _postprocess_standard(
        self,
        outputs: dict[str, npt.NDArray[np.float32]],
        scale: float,
        pad_offset: tuple[int, int],
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[Detection]:
        output_key = next(iter(outputs))
        raw = outputs[output_key]

        if raw.ndim == 3:
            raw = raw[0]

        if raw.shape[0] > raw.shape[1]:
            raw = raw.T

        boxes_xywh = raw[:, :4]
        class_scores = raw[:, 4:]

        max_class_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        mask = max_class_scores >= conf_threshold
        filtered_boxes_xywh = boxes_xywh[mask]
        filtered_scores = max_class_scores[mask]
        filtered_class_ids = class_ids[mask]

        if len(filtered_scores) == 0:
            return []

        boxes_xyxy = self._xywh_to_xyxy(filtered_boxes_xywh)

        keep = _nms_numpy(boxes_xyxy, filtered_scores, iou_threshold)

        pad_left, pad_top = pad_offset
        detections = [
            Detection(
                x1=(float(boxes_xyxy[idx, 0]) - pad_left) / scale,
                y1=(float(boxes_xyxy[idx, 1]) - pad_top) / scale,
                x2=(float(boxes_xyxy[idx, 2]) - pad_left) / scale,
                y2=(float(boxes_xyxy[idx, 3]) - pad_top) / scale,
                confidence=float(filtered_scores[idx]),
                class_id=int(filtered_class_ids[idx]),
            )
            for idx in keep
        ]

        logger.debug("Standard postprocess: {} detections after NMS (conf >= {})", len(detections), conf_threshold)
        return detections

    @staticmethod
    def _xywh_to_xyxy(boxes: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        result = np.empty_like(boxes)
        half_w = boxes[:, 2] / 2.0
        half_h = boxes[:, 3] / 2.0
        result[:, 0] = boxes[:, 0] - half_w
        result[:, 1] = boxes[:, 1] - half_h
        result[:, 2] = boxes[:, 0] + half_w
        result[:, 3] = boxes[:, 1] + half_h
        return result
