"""Continuously stream images from a folder to Redis at a fixed frame rate.

Reads JPEG/PNG files from a directory and circularly pushes them to a
Redis Stream at the specified FPS. Runs indefinitely until interrupted
with Ctrl+C.

Usage:
    uv run python examples/yolo_detection_pipeline/stream_images.py --image-dir ./images --fps 30
    uv run python examples/yolo_detection_pipeline/stream_images.py --fps 10 --stream test:video
    uv run python examples/yolo_detection_pipeline/stream_images.py --fps 1 --max-images 5
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

from loguru import logger

if TYPE_CHECKING:
    from types import FrameType
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

    logger.info("Loaded {} images into memory from {}", len(images), image_dir)
    return images


def stream_images(  # noqa: PLR0913
    image_dir: Path,
    stream_name: str,
    redis_host: str,
    redis_port: int,
    fps: float,
    max_images: int | None = None,
) -> None:
    """Circularly stream images to Redis at the given frame rate."""
    client: Redis[bytes] = Redis(host=redis_host, port=redis_port, db=0, decode_responses=False)
    client.ping()
    logger.info("Connected to Redis at {}:{}", redis_host, redis_port)

    images = load_images(image_dir, max_images)
    frame_interval = 1.0 / fps
    total_sent = 0

    logger.info("Streaming at {:.1f} FPS (interval {:.3f}s) to stream '{}'", fps, frame_interval, stream_name)
    logger.info("Press Ctrl+C to stop")

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
        total_sent += 1

        if total_sent % 100 == 0:
            stream_len = client.xlen(stream_name)
            logger.info("Sent {} frames | stream length: {}", total_sent, stream_len)

        elapsed = time.monotonic() - frame_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    logger.info("Stopped after {} frames", total_sent)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Circularly stream images to Redis at a fixed frame rate")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Image directory")
    parser.add_argument("--stream", default=DEFAULT_STREAM, help=f"Target stream (default: {DEFAULT_STREAM})")
    parser.add_argument("--fps", type=float, required=True, help="Target frame rate (frames per second)")
    parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST, help="Redis host")
    parser.add_argument("--redis-port", type=int, default=DEFAULT_REDIS_PORT, help="Redis port")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images from folder")
    args = parser.parse_args()

    if args.fps <= 0:
        logger.error("FPS must be positive, got {}", args.fps)
        sys.exit(1)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    stream_images(
        image_dir=args.image_dir,
        stream_name=args.stream,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        fps=args.fps,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
