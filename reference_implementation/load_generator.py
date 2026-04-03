"""Load generator: continuously pushes sensor events to 4 Redis streams.

Standalone script using raw redis-py. No typed-flow dependency.
Uses Redis pipelining in batches for high throughput.
Runs until SIGINT/SIGTERM, respecting an optional max rate.

Usage:
    uv run python reference_implementation/load_generator.py --rate 5000
    uv run python reference_implementation/load_generator.py --rate 0 --maxlen 50000
"""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time
from typing import TYPE_CHECKING

from loguru import logger
from redis import Redis

if TYPE_CHECKING:
    import types

STREAMS: list[str] = [
    "sensor:temperature",
    "sensor:humidity",
    "sensor:pressure",
    "sensor:wind",
]

LOCATIONS = ["roof", "basement", "lab-A", "lab-B", "outdoor"]
ZONES = ["north", "south", "east", "west", "center"]
DEFAULT_MAXLEN = 100_000
PIPELINE_BATCH = 500
STATS_INTERVAL_S = 10.0


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
    parser = argparse.ArgumentParser(description="Continuous multi-stream sensor load generator")
    parser.add_argument("--rate", type=float, default=0.0, help="Max events/sec total, 0=unlimited")
    parser.add_argument("--batch", type=int, default=PIPELINE_BATCH, help="Redis pipeline batch size")
    parser.add_argument("--maxlen", type=int, default=DEFAULT_MAXLEN, help="Max stream length per stream")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    return parser.parse_args()


def run_generator(args: argparse.Namespace) -> None:
    """Continuously push events to all 4 sensor Redis streams until shutdown."""
    client: Redis[bytes] = Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=0,
        decode_responses=False,
    )
    client.ping()

    for stream in STREAMS:
        client.delete(stream)
    logger.info("Cleared all input streams")

    rate_desc = f"{args.rate:.0f} events/sec" if args.rate > 0 else "unlimited"
    logger.info(
        "Connected to Redis at {}:{}. Generating to {} (rate={}, maxlen={}, batch={})",
        args.redis_host,
        args.redis_port,
        STREAMS,
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

    events_per_batch = args.batch * len(STREAMS)
    target_batch_time = events_per_batch / args.rate if args.rate > 0 else 0.0

    start = time.monotonic()
    total_pushed = 0
    seq = 0
    prev_stats_time = start
    prev_stats_count = 0

    while not shutdown.is_set():
        batch_start = time.monotonic()
        pipe = client.pipeline(transaction=False)
        for i in range(seq, seq + args.batch):
            for stream, builder in zip(STREAMS, FIELD_BUILDERS, strict=True):
                pipe.xadd(stream, builder(i), maxlen=args.maxlen, approximate=True)
        pipe.execute()
        seq += args.batch
        total_pushed += events_per_batch

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
        "Stopped: {} events across {} streams in {:.2f}s ({:.0f} events/sec)",
        total_pushed,
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
