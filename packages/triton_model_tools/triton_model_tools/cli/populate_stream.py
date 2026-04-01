"""Populate a Redis stream with image events for testing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import redis as redis_lib
from loguru import logger

if TYPE_CHECKING:
    import argparse

    import redis

    from triton_model_tools.cli._types import SubParsers

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})

_DEFAULT_METAL_TYPES = ("steel", "copper", "aluminum")
_DEFAULT_CAMERA_COUNT = 3
_DEFAULT_REDIS_URL = "redis://localhost:6379"
_XADD_MAXLEN = 50_000


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``populate-stream`` subcommand."""
    parser = subparsers.add_parser(
        "populate-stream",
        help="Populate a Redis stream with image events for testing",
    )
    parser.add_argument(
        "--stream",
        type=str,
        required=True,
        help="Redis stream name to populate",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to push",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=_DEFAULT_REDIS_URL,
        help=f"Redis URL (default: {_DEFAULT_REDIS_URL})",
    )
    parser.add_argument(
        "--event-type",
        type=str,
        default="image",
        help="Event type label stored in each message (default: image)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Limit number of images to push (default: all images in directory)",
    )
    parser.add_argument(
        "--camera-count",
        type=int,
        default=_DEFAULT_CAMERA_COUNT,
        help=f"Number of camera IDs to cycle through (default: {_DEFAULT_CAMERA_COUNT})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Read back pushed messages and verify field completeness",
    )
    parser.set_defaults(func=run)


def _collect_images(image_dir: Path, max_count: int | None) -> list[Path]:
    paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS)
    if not paths:
        msg = f"No image files found in {image_dir}"
        raise ValueError(msg)
    if max_count is not None:
        paths = paths[:max_count]
    return paths


def _push_images(
    redis_client: redis.Redis[bytes],
    stream: str,
    image_paths: list[Path],
    event_type: str,
    camera_count: int,
) -> list[str]:
    message_ids: list[str] = []
    metal_types = _DEFAULT_METAL_TYPES

    for idx, image_path in enumerate(image_paths):
        camera_no = str((idx % camera_count) + 1)
        metal_type = metal_types[idx % len(metal_types)]
        frame_counter = str(idx + 1)
        receipt_ts = str(time.time())

        raw_id: Any = redis_client.xadd(
            name=stream,
            fields={
                b"camera_no": camera_no.encode(),
                b"image_path": str(image_path.resolve()).encode(),
                b"frame_counter": frame_counter.encode(),
                b"receipt_ts": receipt_ts.encode(),
                b"metal_type": metal_type.encode(),
                b"event_type": event_type.encode(),
            },
            maxlen=_XADD_MAXLEN,
            approximate=True,
        )
        decoded_id: str = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
        message_ids.append(decoded_id)

        logger.debug(
            "XADD {} -> {}: camera={} frame={} metal={}",
            image_path.name,
            decoded_id,
            camera_no,
            frame_counter,
            metal_type,
        )

    return message_ids


_EXPECTED_FIELDS = frozenset(
    {
        b"camera_no",
        b"image_path",
        b"frame_counter",
        b"receipt_ts",
        b"metal_type",
        b"event_type",
    }
)


def _verify_messages(
    redis_client: redis.Redis[bytes],
    stream: str,
    message_ids: list[str],
) -> bool:
    logger.info("Verifying {} messages in stream {}", len(message_ids), stream)

    raw_messages: Any = redis_client.xrange(stream, min=message_ids[0], max=message_ids[-1])
    pushed_ids = set(message_ids)
    verified_count = 0
    missing_fields_count = 0

    for msg_id_bytes, fields in raw_messages:
        msg_id: str = msg_id_bytes.decode() if isinstance(msg_id_bytes, bytes) else str(msg_id_bytes)
        if msg_id not in pushed_ids:
            continue
        verified_count += 1
        present_fields = set(fields.keys())
        missing = _EXPECTED_FIELDS - present_fields
        if missing:
            missing_fields_count += 1
            missing_names = [f.decode() for f in missing]
            logger.warning("Message {} missing fields: {}", msg_id, missing_names)

    all_ok = verified_count == len(message_ids) and missing_fields_count == 0
    if all_ok:
        logger.info("Verification passed: all {} messages have complete fields", verified_count)
    else:
        logger.error(
            "Verification issues: {}/{} messages found, {} with missing fields",
            verified_count,
            len(message_ids),
            missing_fields_count,
        )
    return all_ok


def run(args: argparse.Namespace) -> None:
    """Execute the ``populate-stream`` subcommand."""
    stream: str = args.stream
    image_dir: Path = args.image_dir.resolve()
    redis_url: str = args.redis_url
    event_type: str = args.event_type
    max_count: int | None = args.count
    camera_count: int = args.camera_count
    verify: bool = args.verify

    logger.info("Populate Stream Tool")
    logger.info("Stream: {}", stream)
    logger.info("Image dir: {}", image_dir)
    logger.info("Redis: {}", redis_url)

    image_paths = _collect_images(image_dir, max_count)
    logger.info("Found {} images to push", len(image_paths))

    redis_client: redis.Redis[bytes] = redis_lib.Redis.from_url(
        redis_url,
        decode_responses=False,
    )

    message_ids = _push_images(redis_client, stream, image_paths, event_type, camera_count)
    logger.info("Pushed {} events to stream {}", len(message_ids), stream)

    if verify:
        _verify_messages(redis_client, stream, message_ids)

    redis_client.close()
