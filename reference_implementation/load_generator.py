"""Camera image load generator: pushes camera events to Redis stream.

Reads JPEG images from a directory and cycles through cam_ids 1-6.

Usage:
    uv run python reference_implementation/load_generator.py --images-dir /path/to/images
    uv run python reference_implementation/load_generator.py --images-dir /path/to/images --rate 100
"""

import argparse
from pathlib import Path

from loguru import logger
from redis import Redis
from triton_model_tools.testing.generators import CameraEventGenerator
from typed_flow.testing.app import StreamPopulatorApp, StreamPopulatorConfig

STREAM = "camera:input"
DEFAULT_MAXLEN = 100_000
DEFAULT_BATCH = 100
NUM_CAMS = 6


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the camera load generator."""
    parser = argparse.ArgumentParser(description="Camera image load generator for CV pipeline")
    parser.add_argument("--images-dir", required=True, help="Directory with JPEG images")
    parser.add_argument("--stream", default=STREAM, help="Redis stream name")
    parser.add_argument("--rate", type=float, default=0.0, help="Max events/sec, 0=unlimited")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Redis pipeline batch size")
    parser.add_argument("--maxlen", type=int, default=DEFAULT_MAXLEN, help="Max stream length")
    parser.add_argument("--num-cams", type=int, default=NUM_CAMS, help="Number of camera IDs to cycle")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    return parser.parse_args()


def main() -> None:
    """Entry point for the camera load generator."""
    args = parse_args()

    client: Redis[bytes] = Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=0,
        decode_responses=False,
    )
    client.ping()
    client.delete(args.stream)

    generator = CameraEventGenerator(
        images_dir=Path(args.images_dir),
        num_cams=args.num_cams,
        stream_name=args.stream,
    )

    config = StreamPopulatorConfig(
        rate=args.rate,
        batch_size=args.batch,
        maxlen=args.maxlen,
    )

    logger.info(
        "Connected to Redis at {}:{}. Generating to {} (rate={}, maxlen={}, batch={})",
        args.redis_host,
        args.redis_port,
        args.stream,
        f"{args.rate:.0f} events/sec" if args.rate > 0 else "unlimited",
        args.maxlen,
        args.batch,
    )

    app = StreamPopulatorApp(
        redis_client=client,
        generators=[generator],
        config=config,
    )
    app.run()
    client.close()


if __name__ == "__main__":
    main()
