"""Run E2E test: push images to Redis, connector processes via Triton, collect results."""

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

from triton_model_tools.cli._types import parse_image_size

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})

_DEFAULT_METAL_TYPES = ("steel", "copper", "aluminum")
_DEFAULT_CAMERA_COUNT = 3
_XADD_MAXLEN = 50_000

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

    populate_group = e2e_parser.add_argument_group("populate-only options")
    populate_group.add_argument(
        "--populate-only",
        action="store_true",
        default=False,
        help="Only push images to Redis stream without running the full E2E pipeline",
    )
    populate_group.add_argument(
        "--event-type",
        type=str,
        default="image",
        help="Event type label stored in each message (default: image)",
    )
    populate_group.add_argument(
        "--count",
        type=int,
        default=None,
        help="Limit number of images to push (default: all images in directory)",
    )
    populate_group.add_argument(
        "--camera-count",
        type=int,
        default=_DEFAULT_CAMERA_COUNT,
        help="Number of camera IDs to cycle through (default: 3)",
    )
    populate_group.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Read back pushed messages and verify field completeness",
    )
    e2e_parser.set_defaults(func=run)


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


def _run_populate_only(args: argparse.Namespace) -> None:
    image_dir: Path = args.image_dir.resolve()
    redis_url: str = args.redis_url
    stream: str = args.input_stream
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


def _run_e2e_test(args: argparse.Namespace) -> None:
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


def run(args: argparse.Namespace) -> None:
    """Execute the ``e2e-test`` subcommand."""
    populate_only: bool = args.populate_only
    if populate_only:
        _run_populate_only(args)
    else:
        _run_e2e_test(args)
