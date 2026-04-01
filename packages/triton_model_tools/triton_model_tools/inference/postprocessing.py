"""YOLO postprocessing: NMS and confidence thresholding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from triton_shared_models.nms import nms

from triton_model_tools.models.detection import Detection

if TYPE_CHECKING:
    import numpy.typing as npt

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45


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

        if raw.shape[0] < raw.shape[1]:
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

        keep = nms(boxes_xyxy, filtered_scores, iou_threshold)

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
