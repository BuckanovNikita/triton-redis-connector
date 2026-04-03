"""Verify pipeline results: compare input vs output stream counts, check metrics.

Usage:
    uv run python reference_implementation/verify.py
    uv run python reference_implementation/verify.py --input-streams sensor:temperature sensor:humidity
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger
from redis import Redis

INPUT_STREAMS = [
    "sensor:temperature",
    "sensor:humidity",
    "sensor:pressure",
    "sensor:wind",
]
OUTPUT_STREAM = "sensor:enriched"
METRICS_STREAM = "metrics:reference"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the verifier."""
    parser = argparse.ArgumentParser(description="Verify enrichment pipeline results")
    parser.add_argument("--input-streams", nargs="+", default=INPUT_STREAMS)
    parser.add_argument("--output-stream", default=OUTPUT_STREAM)
    parser.add_argument("--metrics-stream", default=METRICS_STREAM)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    return parser.parse_args()


def extract_timestamp_ms(message_id: str) -> int:
    """Extract millisecond timestamp from a Redis stream message ID (format: <ms>-<seq>)."""
    return int(message_id.split("-", maxsplit=1)[0])


def _decode_id(raw_id: bytes | str) -> str:
    if isinstance(raw_id, bytes):
        return raw_id.decode()
    return raw_id


def verify_data_streams(
    client: Redis[bytes],
    input_streams: list[str],
    output_stream: str,
) -> int:
    """Verify input vs output event counts. Returns number of missing events."""
    total_input = 0
    for stream in input_streams:
        count = client.xlen(stream)
        logger.info("  {} : {} events", stream, count)
        total_input += count

    output_len = client.xlen(output_stream)
    logger.info("Output stream ({}): {} events", output_stream, output_len)

    if total_input == 0:
        logger.warning("All input streams are empty")
        return 0

    missing = total_input - output_len
    if missing > 0:
        logger.error("MISSING {} events ({:.2f}% loss)", missing, 100.0 * missing / total_input)
    elif missing < 0:
        logger.warning("Output has {} MORE events than input (duplicates?)", abs(missing))
    else:
        logger.info("OK: total input == output, zero loss")

    if output_len > 0:
        first_entries = client.xrange(output_stream, count=1)
        last_entries = client.xrevrange(output_stream, count=1)

        if first_entries and last_entries:
            first_id = _decode_id(first_entries[0][0])
            last_id = _decode_id(last_entries[0][0])

            first_ms = extract_timestamp_ms(first_id)
            last_ms = extract_timestamp_ms(last_id)
            duration_s = (last_ms - first_ms) / 1000.0

            if duration_s > 0:
                throughput = output_len / duration_s
                logger.info(
                    "Throughput: {:.0f} events/sec over {:.2f}s",
                    throughput,
                    duration_s,
                )

    return max(missing, 0)


def verify_metrics_stream(client: Redis[bytes], metrics_stream: str) -> None:
    """Check that metrics stream has events and report zero-value counters."""
    metrics_len = client.xlen(metrics_stream)
    logger.info("Metrics stream ({}): {} events", metrics_stream, metrics_len)

    if metrics_len == 0:
        logger.warning("Metrics stream is empty — MetricEmitter may not be configured")
        return

    sample = client.xrange(metrics_stream, count=20)
    measures_seen: set[str] = set()
    zero_value_measures: set[str] = set()

    for _msg_id, fields in sample:
        measure = fields.get(b"measure", b"").decode()
        value = fields.get(b"value", b"").decode()
        measures_seen.add(measure)
        if value == "0.0":
            zero_value_measures.add(measure)

    logger.info("Sample measures (first 20): {}", sorted(measures_seen))

    if zero_value_measures:
        logger.info("Zero-value metrics found (idle counters): {}", sorted(zero_value_measures))
    else:
        logger.info("No zero-value metrics in sample — pipeline may be actively processing")

    last_entries = client.xrevrange(metrics_stream, count=10)
    logger.info("Last {} metric events:", len(last_entries))
    for msg_id, fields in last_entries:
        decoded = {k.decode(): v.decode() for k, v in fields.items()}
        logger.info("  {} : {}", _decode_id(msg_id), decoded)


def run_verify(args: argparse.Namespace) -> None:
    """Run all verifications."""
    client: Redis[bytes] = Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=0,
        decode_responses=False,
    )
    client.ping()

    logger.info("=== Data Stream Verification ===")
    missing = verify_data_streams(client, args.input_streams, args.output_stream)

    logger.info("=== Metrics Stream Verification ===")
    verify_metrics_stream(client, args.metrics_stream)

    client.close()

    if missing > 0:
        sys.exit(1)


def main() -> None:
    """Entry point for the verifier."""
    args = parse_args()
    run_verify(args)


if __name__ == "__main__":
    main()
