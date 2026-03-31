"""Run E2E test: push images to Redis, connector processes via Triton, collect results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers

from triton_model_tools.cli._types import parse_image_size


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``e2e-test`` subcommand."""
    e2e_parser = subparsers.add_parser(
        "e2e-test",
        help="Run E2E test: push images to Redis, connector processes via Triton, collect results",
    )
    e2e_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to push through the connector",
    )
    e2e_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Triton model name (e.g. yolo11s)",
    )
    e2e_parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://redis:6379",
        help="Redis URL (default: redis://redis:6379)",
    )
    e2e_parser.add_argument(
        "--input-stream",
        type=str,
        default="camera_images",
        help="Redis input stream name (default: camera_images)",
    )
    e2e_parser.add_argument(
        "--output-stream",
        type=str,
        default="detection_results",
        help="Redis output stream name (default: detection_results)",
    )
    e2e_parser.add_argument(
        "--consumer-group",
        type=str,
        default="e2e-test",
        help="Redis consumer group name (default: e2e-test)",
    )
    e2e_parser.add_argument(
        "--target-size",
        type=parse_image_size,
        default=640,
        help="Model input size: int or HxW (e.g. 640 or 960x600)",
    )
    e2e_parser.add_argument(
        "--end2end",
        action="store_true",
        default=False,
        help="Model uses end2end NMS output format",
    )
    e2e_parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for collecting results (default: 30.0)",
    )
    e2e_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``e2e-test`` subcommand."""
    from triton_model_tools.testing.e2e_test import (  # noqa: PLC0415
        ConnectorE2ETestHarness,
        E2ETestConfig,
    )

    image_dir: Path = args.image_dir.resolve()
    model_name: str = args.model
    redis_url: str = args.redis_url
    input_stream: str = args.input_stream
    output_stream: str = args.output_stream
    consumer_group: str = args.consumer_group
    target_size: int | tuple[int, int] = args.target_size
    end2end: bool = args.end2end
    timeout_s: float = args.timeout

    logger.info("E2E Connector Test")
    logger.info("Image dir: {}", image_dir)
    logger.info("Model: {}", model_name)
    logger.info("Redis: {}", redis_url)
    logger.info("Streams: {} -> {}", input_stream, output_stream)

    config = E2ETestConfig(
        redis_url=redis_url,
        input_stream=input_stream,
        output_stream=output_stream,
        model_name=model_name,
        consumer_group=consumer_group,
        target_size=target_size,
        end2end=end2end,
        timeout_s=timeout_s,
    )

    harness = ConnectorE2ETestHarness(config)
    result = harness.run_through_connector(image_dir)

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
        for det in img_result.detections:
            logger.info(
                "    class={} conf={:.3f} bbox=({:.0f},{:.0f},{:.0f},{:.0f})",
                det.class_id,
                det.confidence,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
            )
