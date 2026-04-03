"""Long-running Redis stream population test.

Pushes images from a directory to a Redis input stream at a controlled rate,
cycling through available images if the test duration exceeds the image count.

Usage:
    uv run python examples/long_test/run_long_test.py \
        --duration-minutes 5 \
        --images-per-second 10 \
        --image-dir /workspace/benchmarks/data/coco_val2017 \
        --model-name yolo11s \
        --redis-url redis://redis:6379 \
        --input-stream camera_images
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import redis
from loguru import logger
from tqdm import tqdm
from triton_model_tools.testing.stream_populator import InspectionStreamEntry, populate_stream

if TYPE_CHECKING:
    from collections.abc import Iterator

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
XADD_MAXLEN = 100_000
METAL_TYPES = ("steel", "copper", "aluminum")
DEFAULT_CAMERA_COUNT = 3


def collect_images(image_dir: Path) -> list[Path]:
    """Collect and sort image files from a directory."""
    paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        logger.error("No image files found in {}", image_dir)
        sys.exit(1)
    return paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the long-running test."""
    parser = argparse.ArgumentParser(
        description="Long-running Redis stream population test",
    )
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=5.0,
        help="How long to run the test in minutes (default: 5)",
    )
    parser.add_argument(
        "--images-per-second",
        type=float,
        default=10.0,
        help="Target push rate in images/second (default: 10)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to push",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolo11s",
        help="Model name stored in event metadata (default: yolo11s)",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://redis:6379",
        help="Redis connection URL (default: redis://redis:6379)",
    )
    parser.add_argument(
        "--input-stream",
        type=str,
        default="camera_images",
        help="Redis stream name to push images into (default: camera_images)",
    )
    parser.add_argument(
        "--camera-count",
        type=int,
        default=DEFAULT_CAMERA_COUNT,
        help="Number of simulated cameras to cycle through (default: 3)",
    )
    return parser.parse_args()


def _rate_limited_entries(
    image_paths: list[Path],
    model_name: str,
    camera_count: int,
    duration_s: float,
    target_interval_s: float,
) -> Iterator[InspectionStreamEntry]:
    """Yield inspection entries at a controlled rate for the given duration."""
    image_cycle = itertools.cycle(image_paths)
    start_time = time.monotonic()
    total_pushed = 0

    while time.monotonic() - start_time < duration_s:
        image_path = next(image_cycle)
        yield InspectionStreamEntry(
            camera_no=(total_pushed % camera_count) + 1,
            image_path=str(image_path),
            frame_counter=total_pushed + 1,
            receipt_ts=time.time(),
            metal_type=METAL_TYPES[total_pushed % len(METAL_TYPES)],
            model_name=model_name,
        )

        total_pushed += 1
        expected_time = total_pushed * target_interval_s
        actual_elapsed = time.monotonic() - start_time
        sleep_time = expected_time - actual_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def run_long_test(args: argparse.Namespace) -> None:
    """Run the long-running stream population test."""
    image_dir = args.image_dir.resolve()
    duration_s = args.duration_minutes * 60.0
    target_interval_s = 1.0 / args.images_per_second
    stream_name = args.input_stream

    logger.info("Long Test Configuration:")
    logger.info("  Duration: {:.1f} minutes", args.duration_minutes)
    logger.info("  Target rate: {:.1f} images/second", args.images_per_second)
    logger.info("  Image dir: {}", image_dir)
    logger.info("  Model: {}", args.model_name)
    logger.info("  Redis: {}", args.redis_url)
    logger.info("  Stream: {}", stream_name)

    image_paths = collect_images(image_dir)
    logger.info("Found {} images (will cycle if needed)", len(image_paths))

    redis_client: redis.Redis[bytes] = redis.Redis.from_url(
        args.redis_url,
        decode_responses=False,
    )
    redis_client.ping()
    logger.info("Redis connection established")

    total_expected = int(duration_s * args.images_per_second)
    gen = _rate_limited_entries(image_paths, args.model_name, args.camera_count, duration_s, target_interval_s)
    progress = tqdm(gen, total=total_expected, desc="Pushing images", unit="img", dynamic_ncols=True)

    start_time = time.monotonic()
    try:
        result = populate_stream(redis_client, stream_name, progress, maxlen=XADD_MAXLEN)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        result = None
    finally:
        progress.close()

    total_elapsed = time.monotonic() - start_time
    total_pushed = result.count if result else 0
    actual_rate = total_pushed / total_elapsed if total_elapsed > 0 else 0.0

    logger.info("Long Test Complete:")
    logger.info("  Total images pushed: {}", total_pushed)
    logger.info("  Total elapsed time: {:.1f}s", total_elapsed)
    logger.info("  Average push rate: {:.1f} images/second", actual_rate)
    logger.info("  Target was: {:.1f} images/second", args.images_per_second)

    redis_client.close()


def main() -> None:
    """Entry point for the long test script."""
    args = parse_args()
    run_long_test(args)


if __name__ == "__main__":
    main()
