"""Verify restreamer results: compare input vs output stream counts and measure throughput.

Usage:
    uv run python reference_implementation/verify.py
    uv run python reference_implementation/verify.py --input-stream restreamer:input --output-stream restreamer:output
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger
from redis import Redis

INPUT_STREAM = "restreamer:input"
OUTPUT_STREAM = "restreamer:output"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the verifier."""
    parser = argparse.ArgumentParser(description="Verify restreamer throughput")
    parser.add_argument("--input-stream", default=INPUT_STREAM)
    parser.add_argument("--output-stream", default=OUTPUT_STREAM)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    return parser.parse_args()


def extract_timestamp_ms(message_id: str) -> int:
    """Extract millisecond timestamp from a Redis stream message ID (format: <ms>-<seq>)."""
    return int(message_id.split("-", maxsplit=1)[0])


def run_verify(args: argparse.Namespace) -> None:
    """Compare input and output streams, report throughput."""
    client: Redis[bytes] = Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=0,
        decode_responses=False,
    )
    client.ping()

    input_len = client.xlen(args.input_stream)
    output_len = client.xlen(args.output_stream)

    logger.info("Input stream  ({}): {} events", args.input_stream, input_len)
    logger.info("Output stream ({}): {} events", args.output_stream, output_len)

    if input_len == 0:
        logger.warning("Input stream is empty — nothing to verify")
        client.close()
        return

    missing = input_len - output_len
    if missing > 0:
        logger.error("MISSING {} events ({:.2f}% loss)", missing, 100.0 * missing / input_len)
    elif missing < 0:
        logger.warning("Output has {} MORE events than input (duplicates?)", abs(missing))
    else:
        logger.info("OK: input == output, zero loss")

    if output_len > 0:
        first_entries = client.xrange(args.output_stream, count=1)
        last_entries = client.xrevrange(args.output_stream, count=1)

        if first_entries and last_entries:
            first_id = first_entries[0][0]
            last_id = last_entries[0][0]
            if isinstance(first_id, bytes):
                first_id = first_id.decode()
            if isinstance(last_id, bytes):
                last_id = last_id.decode()

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
            else:
                logger.info("All output events have the same timestamp — cannot compute throughput")

    client.close()

    if missing > 0:
        sys.exit(1)


def main() -> None:
    """Entry point for the verifier."""
    args = parse_args()
    run_verify(args)


if __name__ == "__main__":
    main()
