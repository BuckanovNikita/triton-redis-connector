"""Run ultralytics YOLO prediction on an image folder and save results as txt files."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from ultralytics import YOLO  # type: ignore[attr-defined]

from triton_model_tools.inference.detection_io import collect_image_paths, save_detections_to_txt
from triton_model_tools.models.detection import Detection, FolderInferenceResult, ImageDetectionResult

if TYPE_CHECKING:
    from pathlib import Path

MILLISECONDS_PER_SECOND = 1000.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45


class PredictConfig(BaseModel):
    """Configuration for ultralytics YOLO prediction."""

    model_config = ConfigDict(frozen=True)

    model_path: str = "yolo11s.pt"
    conf_threshold: float = Field(default=DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=DEFAULT_IOU_THRESHOLD, ge=0.0, le=1.0)
    target_size: int = Field(default=640, gt=0)


def run_ultralytics_predict(
    image_dir: Path,
    output_dir: Path,
    config: PredictConfig,
) -> FolderInferenceResult:
    """Run ultralytics model.predict() on all images and save detection txt files.

    Args:
        image_dir: Directory containing images.
        output_dir: Directory where txt result files will be written.
        config: Prediction configuration.

    Returns:
        FolderInferenceResult with all detection results and timing.

    """
    image_paths = collect_image_paths(image_dir)
    logger.info(
        "Running ultralytics predict: model={} images={} output={}",
        config.model_path,
        len(image_paths),
        output_dir,
    )

    model: Any = YOLO(config.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_list: list[ImageDetectionResult] = []
    total_start = time.perf_counter()

    for image_path in image_paths:
        start = time.perf_counter()
        predictions = model.predict(
            str(image_path),
            conf=config.conf_threshold,
            iou=config.iou_threshold,
            imgsz=config.target_size,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND

        detections: list[Detection] = []
        for result in predictions:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i]
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    detections.append(
                        Detection(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                            confidence=conf,
                            class_id=cls_id,
                        ),
                    )

        txt_path = output_dir / f"{image_path.stem}.txt"
        save_detections_to_txt(detections, txt_path)

        img_result = ImageDetectionResult(
            image_path=image_path.stem,
            model_name=config.model_path,
            detections=detections,
            inference_time_ms=elapsed_ms,
        )
        results_list.append(img_result)

        logger.debug(
            "  {}: {} detections ({:.1f}ms)",
            image_path.name,
            len(detections),
            elapsed_ms,
        )

    total_time_s = time.perf_counter() - total_start
    avg_time_ms = (total_time_s * MILLISECONDS_PER_SECOND / len(results_list)) if results_list else 0.0

    logger.info(
        "Ultralytics predict complete: {} images, {:.2f}s total, {:.1f}ms avg",
        len(results_list),
        total_time_s,
        avg_time_ms,
    )

    return FolderInferenceResult(
        model_name=config.model_path,
        results=results_list,
        total_time_s=total_time_s,
        avg_time_ms=avg_time_ms,
    )
