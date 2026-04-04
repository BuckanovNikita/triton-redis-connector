"""Camera image load generator: pushes camera events to Redis stream.

Reads JPEG images from a directory and cycles through cam_ids 1-6.
Cams 1-4 send image_path (filesystem), cams 5-6 send image_bytes (in-memory).

Usage:
    uv run python reference_implementation/load_generator.py --images-dir /path/to/images
    uv run python reference_implementation/load_generator.py --images-dir /path/to/images --rate 100
"""

import argparse
import itertools
import signal
import threading
import time
import types
from pathlib import Path

from loguru import logger
from redis import Redis

STREAM = "camera:input"
DEFAULT_MAXLEN = 100_000
PIPELINE_BATCH = 100
STATS_INTERVAL_S = 10.0
NUM_CAMS = 6


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the camera load generator."""
    parser = argparse.ArgumentParser(description="Camera image load generator for CV pipeline")
    parser.add_argument("--images-dir", required=True, help="Directory with JPEG images")
    parser.add_argument("--stream", default=STREAM, help="Redis stream name")
    parser.add_argument("--rate", type=float, default=0.0, help="Max events/sec, 0=unlimited")
    parser.add_argument("--batch", type=int, default=PIPELINE_BATCH, help="Redis pipeline batch size")
    parser.add_argument("--maxlen", type=int, default=DEFAULT_MAXLEN, help="Max stream length")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    return parser.parse_args()


def _collect_images(images_dir: str) -> list[Path]:
    """Collect JPEG/PNG image paths from a directory."""
    path = Path(images_dir)
    if not path.is_dir():
        msg = f"Images directory does not exist: {images_dir}"
        raise FileNotFoundError(msg)
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(p for p in path.iterdir() if p.suffix.lower() in extensions)
    if not images:
        msg = f"No images found in {images_dir}"
        raise FileNotFoundError(msg)
    return images


def _build_event(image_path: Path, cam_id: int) -> dict[str, str | bytes]:
    """Build a Redis stream event dict for a camera image.

    All cams use image_path (filesystem reference) because typed-flow's Redis
    connector decodes all fields as UTF-8 strings, which is incompatible with
    raw binary image data.
    """
    return {
        "cam_id": str(cam_id),
        "image_path": str(image_path.resolve()),
    }


def run_generator(args: argparse.Namespace) -> None:
    """Continuously push camera events to Redis stream until shutdown."""
    client: Redis[bytes] = Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=0,
        decode_responses=False,
    )
    client.ping()
    client.delete(args.stream)

    images = _collect_images(args.images_dir)
    image_cycle = itertools.cycle(images)
    cam_cycle = itertools.cycle(range(1, NUM_CAMS + 1))

    rate_desc = f"{args.rate:.0f} events/sec" if args.rate > 0 else "unlimited"
    logger.info(
        "Connected to Redis at {}:{}. Generating to {} ({} images, rate={}, maxlen={}, batch={})",
        args.redis_host,
        args.redis_port,
        args.stream,
        len(images),
        rate_desc,
        args.maxlen,
        args.batch,
    )

    shutdown = threading.Event()

    def handle_signal(signum: int, _frame: types.FrameType | None) -> None:
        logger.info("Received {}, stopping...", signal.Signals(signum).name)
        shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    target_batch_time = args.batch / args.rate if args.rate > 0 else 0.0

    start = time.monotonic()
    total_pushed = 0
    prev_stats_time = start
    prev_stats_count = 0

    while not shutdown.is_set():
        batch_start = time.monotonic()
        pipe = client.pipeline(transaction=False)
        for _ in range(args.batch):
            image = next(image_cycle)
            cam_id = next(cam_cycle)
            fields = _build_event(image, cam_id)
            pipe.xadd(args.stream, fields, maxlen=args.maxlen, approximate=True)
        pipe.execute()
        total_pushed += args.batch

        now = time.monotonic()
        if now - prev_stats_time >= STATS_INTERVAL_S:
            interval_events = total_pushed - prev_stats_count
            interval_rate = interval_events / (now - prev_stats_time)
            logger.info(
                "[LOAD] total={} interval_rate={:.0f} events/sec elapsed={:.1f}s",
                total_pushed,
                interval_rate,
                now - start,
            )
            prev_stats_time = now
            prev_stats_count = total_pushed

        if target_batch_time > 0:
            elapsed_batch = time.monotonic() - batch_start
            sleep_time = target_batch_time - elapsed_batch
            if sleep_time > 0:
                shutdown.wait(timeout=sleep_time)

    total_elapsed = time.monotonic() - start
    actual_rate = total_pushed / total_elapsed if total_elapsed > 0 else 0.0
    logger.info(
        "Stopped: {} events in {:.2f}s ({:.0f} events/sec)",
        total_pushed,
        total_elapsed,
        actual_rate,
    )
    client.close()


def main() -> None:
    """Entry point for the camera load generator."""
    args = parse_args()
    run_generator(args)


if __name__ == "__main__":
    main()
