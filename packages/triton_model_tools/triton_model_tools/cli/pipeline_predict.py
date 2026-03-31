"""Run images through Redis pipeline and save detection txt files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``pipeline-predict`` subcommand."""
    pp_parser = subparsers.add_parser(
        "pipeline-predict",
        help="Run images through Redis pipeline and save detection txt files",
    )
    pp_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to push through the pipeline",
    )
    pp_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save detection txt files",
    )
    pp_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Triton model name (e.g. yolo11s)",
    )
    pp_parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://redis:6379",
        help="Redis URL (default: redis://redis:6379)",
    )
    pp_parser.add_argument(
        "--input-stream",
        type=str,
        default="camera_images",
        help="Redis input stream name (default: camera_images)",
    )
    pp_parser.add_argument(
        "--output-stream",
        type=str,
        default="detection_results",
        help="Redis output stream name (default: detection_results)",
    )
    pp_parser.add_argument(
        "--consumer-group",
        type=str,
        default="pipeline-predict",
        help="Redis consumer group name (default: pipeline-predict)",
    )
    pp_parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for collecting results (default: 30.0)",
    )
    pp_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``pipeline-predict`` subcommand."""
    from triton_model_tools.testing.e2e_test import E2ETestConfig  # noqa: PLC0415
    from triton_model_tools.testing.pipeline_predict import run_pipeline_predict  # noqa: PLC0415

    image_dir: Path = args.image_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    model_name: str = args.model
    redis_url: str = args.redis_url
    input_stream: str = args.input_stream
    output_stream: str = args.output_stream
    consumer_group: str = args.consumer_group
    timeout_s: float = args.timeout

    logger.info("Pipeline Prediction")
    logger.info("Image dir: {}", image_dir)
    logger.info("Output dir: {}", output_dir)
    logger.info("Model: {}", model_name)
    logger.info("Redis: {}", redis_url)

    config = E2ETestConfig(
        redis_url=redis_url,
        input_stream=input_stream,
        output_stream=output_stream,
        model_name=model_name,
        consumer_group=consumer_group,
        timeout_s=timeout_s,
    )

    result = run_pipeline_predict(image_dir, output_dir, config)

    logger.info("Results: {} images processed", len(result.results))
    logger.info("Total time: {:.2f}s", result.total_time_s)
    logger.info("Average time: {:.1f}ms per image", result.avg_time_ms)
