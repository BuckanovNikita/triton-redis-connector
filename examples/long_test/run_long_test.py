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

import redis
from loguru import logger
from tqdm import tqdm

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
XADD_MAXLEN = 100_000
LOG_INTERVAL_S = 10.0
METAL_TYPES = ("steel", "copper", "aluminum")
DEFAULT_CAMERA_COUNT = 3


def collect_images(image_dir: Path) -> list[Path]:
    paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        logger.error("No image files found in {}", image_dir)
        sys.exit(1)
    return paths


def parse_args() -> argparse.Namespace:
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


def run_long_test(args: argparse.Namespace) -> None:
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
    total_pushed = 0
    start_time = time.monotonic()
    last_log_time = start_time
    interval_pushed = 0

    image_cycle = itertools.cycle(enumerate(image_paths))
    progress = tqdm(
        total=total_expected,
        desc="Pushing images",
        unit="img",
        dynamic_ncols=True,
    )

    try:
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= duration_s:
                break

            cycle_idx, image_path = next(image_cycle)
            camera_no = str((total_pushed % args.camera_count) + 1)
            metal_type = METAL_TYPES[total_pushed % len(METAL_TYPES)]

            redis_client.xadd(
                name=stream_name,
                fields={
                    b"camera_no": camera_no.encode(),
                    b"image_path": str(image_path).encode(),
                    b"frame_counter": str(total_pushed + 1).encode(),
                    b"receipt_ts": str(time.time()).encode(),
                    b"metal_type": metal_type.encode(),
                    b"event_type": b"image",
                    b"model_name": args.model_name.encode(),
                },
                maxlen=XADD_MAXLEN,
                approximate=True,
            )
            total_pushed += 1
            interval_pushed += 1
            progress.update(1)

            now = time.monotonic()
            if now - last_log_time >= LOG_INTERVAL_S:
                actual_rate = interval_pushed / (now - last_log_time)
                logger.info(
                    "Progress: {} images pushed, rate: {:.1f} img/s (target: {:.1f})",
                    total_pushed,
                    actual_rate,
                    args.images_per_second,
                )
                last_log_time = now
                interval_pushed = 0

            expected_time = total_pushed * target_interval_s
            actual_elapsed = time.monotonic() - start_time
            sleep_time = expected_time - actual_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        progress.close()

    total_elapsed = time.monotonic() - start_time
    actual_rate = total_pushed / total_elapsed if total_elapsed > 0 else 0.0

    logger.info("Long Test Complete:")
    logger.info("  Total images pushed: {}", total_pushed)
    logger.info("  Total elapsed time: {:.1f}s", total_elapsed)
    logger.info("  Average push rate: {:.1f} images/second", actual_rate)
    logger.info("  Target was: {:.1f} images/second", args.images_per_second)

    redis_client.close()


def main() -> None:
    args = parse_args()
    run_long_test(args)


if __name__ == "__main__":
    main()
