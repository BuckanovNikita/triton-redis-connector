"""Load generator: pushes sensor events to 4 Redis streams at controlled rate.

Standalone script using raw redis-py. No typed-flow dependency.
Uses Redis pipelining in batches for high throughput.

Usage:
    uv run python reference_implementation/load_generator.py --count 10000 --rate 0
    uv run python reference_implementation/load_generator.py --count 5000 --rate 2000
"""

from __future__ import annotations

import argparse
import math
import time

from loguru import logger
from redis import Redis

STREAMS: list[str] = [
    "sensor:temperature",
    "sensor:humidity",
    "sensor:pressure",
    "sensor:wind",
]

LOCATIONS = ["roof", "basement", "lab-A", "lab-B", "outdoor"]
ZONES = ["north", "south", "east", "west", "center"]
XADD_MAXLEN = 100_000
PIPELINE_BATCH = 500


def _temperature_fields(i: int) -> dict[str, str]:
    return {
        "sensor_id": f"temp-{i % 10}",
        "celsius": f"{20.0 + 10.0 * math.sin(i / 50.0):.2f}",
        "location": LOCATIONS[i % len(LOCATIONS)],
    }


def _humidity_fields(i: int) -> dict[str, str]:
    return {
        "sensor_id": f"hum-{i % 8}",
        "humidity_pct": f"{40.0 + 30.0 * math.cos(i / 40.0):.2f}",
        "zone": ZONES[i % len(ZONES)],
    }


def _pressure_fields(i: int) -> dict[str, str]:
    return {
        "sensor_id": f"press-{i % 6}",
        "hpa": f"{1013.25 + 5.0 * math.sin(i / 100.0):.2f}",
        "altitude_m": f"{100.0 + (i % 5) * 50.0:.1f}",
    }


def _wind_fields(i: int) -> dict[str, str]:
    return {
        "sensor_id": f"wind-{i % 4}",
        "speed_mps": f"{0.5 + (i * 7.3 % 24.5):.2f}",
        "direction_deg": str(i * 37 % 360),
        "gust_mps": f"{1.0 + (i * 11.7 % 34.0):.2f}",
    }


FIELD_BUILDERS = [_temperature_fields, _humidity_fields, _pressure_fields, _wind_fields]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the load generator."""
    parser = argparse.ArgumentParser(description="Multi-stream sensor load generator")
    parser.add_argument("--count", type=int, default=10_000, help="Events per stream")
    parser.add_argument("--rate", type=float, default=0.0, help="Events/sec total, 0=unlimited")
    parser.add_argument("--batch", type=int, default=PIPELINE_BATCH, help="Redis pipeline batch size")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    return parser.parse_args()


def run_generator(args: argparse.Namespace) -> None:
    """Push events to all 4 sensor Redis streams using pipelined batches."""
    client: Redis[bytes] = Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=0,
        decode_responses=False,
    )
    client.ping()
    total_events = args.count * len(STREAMS)
    logger.info(
        "Connected to Redis at {}:{}. Pushing {} events/stream ({} total) to {}",
        args.redis_host,
        args.redis_port,
        args.count,
        total_events,
        STREAMS,
    )

    target_interval = 1.0 / args.rate if args.rate > 0 else 0.0
    start = time.monotonic()
    pushed = 0

    while pushed < args.count:
        batch_end = min(pushed + args.batch, args.count)
        pipe = client.pipeline(transaction=False)
        for i in range(pushed, batch_end):
            for stream, builder in zip(STREAMS, FIELD_BUILDERS, strict=True):
                pipe.xadd(stream, builder(i), maxlen=XADD_MAXLEN, approximate=True)
        pipe.execute()
        pushed = batch_end

        if target_interval > 0:
            expected = pushed * len(STREAMS) * target_interval
            elapsed = time.monotonic() - start
            sleep_time = expected - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    total_elapsed = time.monotonic() - start
    actual_rate = total_events / total_elapsed if total_elapsed > 0 else 0.0

    logger.info(
        "Done: {} events across {} streams in {:.2f}s ({:.0f} events/sec)",
        total_events,
        len(STREAMS),
        total_elapsed,
        actual_rate,
    )
    client.close()


def main() -> None:
    """Entry point for the load generator."""
    args = parse_args()
    run_generator(args)


if __name__ == "__main__":
    main()
