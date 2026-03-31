"""Synthetic processors for the camera inference demo.

Generates fake frames and detection results — no real inference.
Useful as a minimal template for building real pipelines.
"""

from __future__ import annotations

import json
import threading
import time

from loguru import logger

from examples.camera_inference.events import CameraFrame, DetectionResult


class CameraSource:
    """Simulates camera capture by generating synthetic frames."""

    def __init__(self, camera_id: str = "cam-1", interval_s: float = 1.0) -> None:  # noqa: D107
        self.stop = threading.Event()
        self._camera_id = camera_id
        self._interval_s = interval_s
        self._frame_counter = 0

    def __call__(self) -> CameraFrame | None:  # noqa: D102
        if self.stop.wait(timeout=self._interval_s):
            return None
        self._frame_counter += 1
        return CameraFrame(
            camera_id=self._camera_id,
            frame_id=self._frame_counter,
            image_key=f"images:{self._camera_id}:{self._frame_counter}",
            timestamp_ms=int(time.time() * 1000),
        )


class FramePreprocessor:
    """Produces synthetic detection results (no real inference)."""

    def __init__(self, model_name: str = "yolo11s") -> None:  # noqa: D107
        self._model_name = model_name

    def __call__(self, event: CameraFrame) -> DetectionResult | None:  # noqa: D102
        detections = [{"class_id": 0, "confidence": 0.95, "bbox": [100, 100, 200, 200]}]
        return DetectionResult(
            camera_id=event.camera_id,
            frame_id=event.frame_id,
            model_name=self._model_name,
            detections_json=json.dumps(detections),
            inference_time_ms=15.0,
        )


class ResultSink:
    """Logs detection results."""

    def __call__(self, event: DetectionResult) -> None:  # noqa: D102
        logger.info(
            "Detection: camera={} frame={} model={} time={:.1f}ms",
            event.camera_id,
            event.frame_id,
            event.model_name,
            event.inference_time_ms,
        )
