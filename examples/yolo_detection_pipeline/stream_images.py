"""Push images from a folder into a Redis Stream.

Two modes:
  - Batch (default): push all images once and exit.
  - Streaming (--fps): circularly push images at a fixed frame rate until Ctrl+C.

Usage:
    # Batch: push all images once
    uv run python examples/yolo_detection_pipeline/stream_images.py

    # Batch: push 5 images, repeat 3 times
    uv run python examples/yolo_detection_pipeline/stream_images.py --max-images 5 --repeat 3

    # Stream: continuously push at 10 FPS
    uv run python examples/yolo_detection_pipeline/stream_images.py --fps 10
"""

from __future__ import annotations

import argparse
import itertools
import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType

from loguru import logger
from redis import Redis

DEFAULT_STREAM = "test:input_images"
DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "data" / "coco8" / "images" / "val"
DEFAULT_REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
DEFAULT_REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
DEFAULT_MAXLEN = 1000

_shutdown_requested = False


def _handle_signal(_signum: int, _frame: FrameType | None) -> None:
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = True
    logger.info("Shutdown requested, finishing current frame...")


def load_images(image_dir: Path, max_images: int | None = None) -> list[tuple[str, bytes]]:
    """Load images from directory into memory. Returns list of (filename, bytes)."""
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        logger.error("No images found in {}", image_dir)
        sys.exit(1)

    if max_images is not None:
        paths = paths[:max_images]

    images = [(p.name, p.read_bytes()) for p in paths]
    logger.info("Loaded {} images from {}", len(images), image_dir)
    return images


def push_images(
    client: Redis[bytes],
    images: list[tuple[str, bytes]],
    stream_name: str,
    repeat: int = 1,
) -> int:
    """Push images to Redis stream in batch mode. Returns total messages sent."""
    total = 0
    for round_num in range(1, repeat + 1):
        if repeat > 1:
            logger.info("--- Round {}/{} ---", round_num, repeat)
        for filename, raw_bytes in images:
            if _shutdown_requested:
                return total
            client.xadd(
                name=stream_name,
                fields={"data": raw_bytes, "filename": filename.encode()},
                maxlen=DEFAULT_MAXLEN,
                approximate=True,
            )
            total += 1
            logger.info("Pushed {} ({:,} bytes) -> {}", filename, len(raw_bytes), stream_name)
    return total


def stream_continuous(
    client: Redis[bytes],
    images: list[tuple[str, bytes]],
    stream_name: str,
    fps: float,
) -> int:
    """Circularly stream images at fixed FPS until shutdown. Returns total sent."""
    frame_interval = 1.0 / fps
    total = 0

    logger.info("Streaming at {:.1f} FPS to '{}'. Ctrl+C to stop.", fps, stream_name)

    for filename, raw_bytes in itertools.cycle(images):
        if _shutdown_requested:
            break

        frame_start = time.monotonic()
        client.xadd(
            name=stream_name,
            fields={"data": raw_bytes, "filename": filename.encode()},
            maxlen=DEFAULT_MAXLEN,
            approximate=True,
        )
        total += 1

        if total % 100 == 0:
            logger.info("Sent {} frames | stream length: {}", total, client.xlen(stream_name))

        elapsed = time.monotonic() - frame_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    return total


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Push images to Redis Stream (batch or continuous)")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Image directory")
    parser.add_argument("--stream", default=DEFAULT_STREAM, help=f"Target stream (default: {DEFAULT_STREAM})")
    parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST, help="Redis host")
    parser.add_argument("--redis-port", type=int, default=DEFAULT_REDIS_PORT, help="Redis port")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images from folder")
    parser.add_argument("--fps", type=float, default=None, help="Continuous streaming at this FPS (omit for batch)")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat N times in batch mode (default: 1)")
    args = parser.parse_args()

    if args.fps is not None and args.fps <= 0:
        logger.error("FPS must be positive, got {}", args.fps)
        sys.exit(1)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    client: Redis[bytes] = Redis(host=args.redis_host, port=args.redis_port, db=0, decode_responses=False)
    client.ping()
    logger.info("Connected to Redis at {}:{}", args.redis_host, args.redis_port)

    images = load_images(args.image_dir, args.max_images)

    if args.fps is not None:
        total = stream_continuous(client, images, args.stream, args.fps)
    else:
        total = push_images(client, images, args.stream, args.repeat)

    stream_len = client.xlen(args.stream)
    logger.info("Done: {} messages sent. Stream '{}' length: {}", total, args.stream, stream_len)


if __name__ == "__main__":
    main()
