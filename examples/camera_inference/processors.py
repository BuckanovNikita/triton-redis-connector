"""Example processors using framework execution variants.

Demonstrates how to create Source, Processor, and Sink implementations
that work with the triton-redis-connector pipeline framework.
"""

from __future__ import annotations

import json
import threading
import time

from loguru import logger

from examples.camera_inference.events import CameraFrame, DetectionResult


class CameraSource:
    """Example ThreadSource that simulates camera frame capture.

    In production, this would read from a Redis Stream or camera API.
    For demonstration, it generates synthetic frames at a fixed interval.
    """

    def __init__(self, camera_id: str, interval_s: float = 1.0) -> None:  # noqa: D107
        self.stop = threading.Event()
        self._camera_id = camera_id
        self._interval_s = interval_s
        self._frame_counter = 0

    def __call__(self) -> CameraFrame | None:  # noqa: D102
        if self.stop.wait(timeout=self._interval_s):
            return None
        self._frame_counter += 1
        frame = CameraFrame(
            camera_id=self._camera_id,
            frame_id=self._frame_counter,
            image_key=f"images:{self._camera_id}:{self._frame_counter}",
            timestamp_ms=int(time.time() * 1000),
        )
        logger.debug("Captured frame {} from camera {}", self._frame_counter, self._camera_id)
        return frame


class FramePreprocessor:
    """Example inline Processor that transforms CameraFrame to DetectionResult.

    In production, this would decode the image, run letterbox preprocessing,
    call Triton for inference, and parse the results. For demonstration,
    it creates a synthetic detection result.
    """

    def __init__(self, model_name: str = "yolo11s") -> None:  # noqa: D107
        self._model_name = model_name

    def __call__(self, event: CameraFrame) -> DetectionResult | None:  # noqa: D102
        detections = [
            {"class_id": 0, "confidence": 0.95, "bbox": [100, 100, 200, 200]},
        ]
        return DetectionResult(
            camera_id=event.camera_id,
            frame_id=event.frame_id,
            model_name=self._model_name,
            detections_json=json.dumps(detections),
            inference_time_ms=15.0,
        )


class ResultSink:
    """Example inline Sink that logs detection results.

    In production, this would write results to a Redis Stream
    or external service.
    """

    def __call__(self, event: DetectionResult) -> None:  # noqa: D102
        logger.info(
            "Detection result: camera={} frame={} model={} time={:.1f}ms",
            event.camera_id,
            event.frame_id,
            event.model_name,
            event.inference_time_ms,
        )
