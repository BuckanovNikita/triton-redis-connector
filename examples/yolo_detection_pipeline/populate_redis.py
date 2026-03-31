"""Push test images into a Redis Stream for integration testing.

Reads JPEG files from benchmarks/data/coco8/images/val/ (or a custom directory)
and pushes them to a Redis Stream as raw bytes. Each message has:
  - data: raw image bytes
  - filename: original filename

Usage:
    uv run python examples/yolo_detection_pipeline/populate_redis.py
    uv run python examples/yolo_detection_pipeline/populate_redis.py --max-images 5
    uv run python examples/yolo_detection_pipeline/populate_redis.py --repeat 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger
from redis import Redis

DEFAULT_STREAM = "test:input_images"
DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "data" / "coco8" / "images" / "val"
DEFAULT_REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
DEFAULT_REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
DEFAULT_MAXLEN = 1000


def populate_redis(  # noqa: PLR0913
    image_dir: Path,
    stream_name: str,
    redis_host: str,
    redis_port: int,
    max_images: int | None = None,
    repeat: int = 1,
) -> list[str]:
    """Read images from directory and push to Redis stream.

    Returns list of all message IDs created.
    """
    client: Redis[bytes] = Redis(host=redis_host, port=redis_port, db=0, decode_responses=False)
    client.ping()  # pyright: ignore[reportUnknownMemberType]
    logger.info("Connected to Redis at {}:{}", redis_host, redis_port)

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        logger.error("No images found in {}", image_dir)
        sys.exit(1)

    if max_images is not None:
        image_paths = image_paths[:max_images]

    logger.info("Found {} images in {}", len(image_paths), image_dir)

    all_message_ids: list[str] = []

    for round_num in range(1, repeat + 1):
        if repeat > 1:
            logger.info("--- Round {}/{} ---", round_num, repeat)

        for path in image_paths:
            raw_bytes = path.read_bytes()
            msg_id_bytes = client.xadd(  # pyright: ignore[reportUnknownMemberType]
                name=stream_name,
                fields={"data": raw_bytes, "filename": path.name.encode()},
                maxlen=DEFAULT_MAXLEN,
                approximate=True,
            )
            msg_id = msg_id_bytes.decode() if isinstance(msg_id_bytes, bytes) else str(msg_id_bytes)
            all_message_ids.append(msg_id)
            logger.info(
                "Pushed {} ({:,} bytes) -> {} as {}",
                path.name,
                len(raw_bytes),
                stream_name,
                msg_id,
            )

    stream_len = client.xlen(stream_name)  # pyright: ignore[reportUnknownMemberType]
    logger.info(
        "Done: pushed {} messages total. Stream {} length: {}",
        len(all_message_ids),
        stream_name,
        stream_len,
    )

    return all_message_ids


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Push test images to Redis stream")
    parser.add_argument("--stream", default=DEFAULT_STREAM, help=f"Target stream name (default: {DEFAULT_STREAM})")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Image directory (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST, help=f"Redis host (default: {DEFAULT_REDIS_HOST})")
    parser.add_argument(
        "--redis-port", type=int, default=DEFAULT_REDIS_PORT, help=f"Redis port (default: {DEFAULT_REDIS_PORT})"
    )
    parser.add_argument("--max-images", type=int, default=None, help="Max number of images to push (default: all)")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat N times for load testing (default: 1)")
    args = parser.parse_args()

    populate_redis(
        image_dir=args.image_dir,
        stream_name=args.stream,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        max_images=args.max_images,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    main()
