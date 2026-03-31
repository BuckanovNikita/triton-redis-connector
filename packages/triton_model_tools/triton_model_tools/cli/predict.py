"""Run ultralytics YOLO prediction on an image folder and save txt results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers

from triton_model_tools.cli._types import parse_image_size


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``predict`` subcommand."""
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run ultralytics YOLO prediction on an image folder and save txt results",
    )
    predict_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to predict on",
    )
    predict_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save detection txt files",
    )
    predict_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Ultralytics model path or name (default: yolo11s.pt)",
    )
    predict_parser.add_argument(
        "--target-size",
        type=parse_image_size,
        default=640,
        help="Model input size: int or HxW (e.g. 640 or 960x600)",
    )
    predict_parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    predict_parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    predict_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``predict`` subcommand."""
    from triton_model_tools.inference.predict import PredictConfig, run_ultralytics_predict  # noqa: PLC0415

    image_dir: Path = args.image_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    model: str = args.model
    target_size: int | tuple[int, int] = args.target_size
    conf_threshold: float = args.conf_threshold
    iou_threshold: float = args.iou_threshold

    logger.info("Ultralytics YOLO Prediction")
    logger.info("Image dir: {}", image_dir)
    logger.info("Output dir: {}", output_dir)
    logger.info("Model: {}", model)

    config = PredictConfig(
        model_path=model,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        target_size=target_size,
    )

    result = run_ultralytics_predict(image_dir, output_dir, config)

    logger.info("Results: {} images processed", len(result.results))
    logger.info("Total time: {:.2f}s", result.total_time_s)
    logger.info("Average time: {:.1f}ms per image", result.avg_time_ms)

    for img_result in result.results:
        logger.info(
            "  {}: {} detections ({:.1f}ms)",
            img_result.image_path,
            len(img_result.detections),
            img_result.inference_time_ms,
        )
