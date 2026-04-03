"""Load generator: pushes SampleEvent entries to a Redis stream at controlled rate.

Standalone script using raw redis-py. No typed-flow dependency.
Uses Redis pipelining in batches for high throughput.

Usage:
    uv run python reference_implementation/load_generator.py --count 10000 --rate 0
    uv run python reference_implementation/load_generator.py --count 50000 --rate 10000
"""

from __future__ import annotations

import argparse
import time

from loguru import logger
from redis import Redis

INPUT_STREAM = "restreamer:input"
XADD_MAXLEN = 100_000
PIPELINE_BATCH = 500


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the load generator."""
    parser = argparse.ArgumentParser(description="Restreamer load generator")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--rate", type=float, default=0.0, help="Events/sec, 0=unlimited")
    parser.add_argument("--batch", type=int, default=PIPELINE_BATCH, help="Redis pipeline batch size")
    parser.add_argument("--stream", default=INPUT_STREAM)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    return parser.parse_args()


def run_generator(args: argparse.Namespace) -> None:
    """Push events to the Redis input stream using pipelined batches."""
    client: Redis[bytes] = Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=0,
        decode_responses=False,
    )
    client.ping()
    logger.info(
        "Connected to Redis at {}:{}. Pushing {} events to {} (batch={})",
        args.redis_host,
        args.redis_port,
        args.count,
        args.stream,
        args.batch,
    )

    target_interval = 1.0 / args.rate if args.rate > 0 else 0.0
    start = time.monotonic()
    pushed = 0

    while pushed < args.count:
        batch_end = min(pushed + args.batch, args.count)
        pipe = client.pipeline(transaction=False)
        for i in range(pushed, batch_end):
            fields: dict[str, str] = {
                "sequence_id": str(i),
                "timestamp": str(time.time()),
                "payload": f"event-{i}",
            }
            pipe.xadd(args.stream, fields, maxlen=XADD_MAXLEN, approximate=True)
        pipe.execute()
        pushed = batch_end

        if target_interval > 0:
            expected = pushed * target_interval
            elapsed = time.monotonic() - start
            sleep_time = expected - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    total_elapsed = time.monotonic() - start
    actual_rate = args.count / total_elapsed if total_elapsed > 0 else 0.0

    logger.info("Done: {} events in {:.2f}s ({:.0f} events/sec)", args.count, total_elapsed, actual_rate)
    client.close()


def main() -> None:
    """Entry point for the load generator."""
    args = parse_args()
    run_generator(args)


if __name__ == "__main__":
    main()
