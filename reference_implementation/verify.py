"""Verify CV pipeline results: check input vs output stream counts, metrics.

Usage:
    uv run python reference_implementation/verify.py
    uv run python reference_implementation/verify.py --input-streams camera:input
"""

import argparse

from loguru import logger
from redis import Redis

INPUT_STREAMS = ["camera:input"]
OUTPUT_STREAMS = ["results:detection", "results:classification"]
METRICS_STREAM = "metrics:cv-pipeline"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the verifier."""
    parser = argparse.ArgumentParser(description="Verify CV inference pipeline results")
    parser.add_argument("--input-streams", nargs="+", default=INPUT_STREAMS)
    parser.add_argument("--output-streams", nargs="+", default=OUTPUT_STREAMS)
    parser.add_argument("--metrics-stream", default=METRICS_STREAM)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    return parser.parse_args()


def _decode_id(raw_id: bytes | str) -> str:
    if isinstance(raw_id, bytes):
        return raw_id.decode()
    return raw_id


def _extract_timestamp_ms(message_id: str) -> int:
    """Extract millisecond timestamp from a Redis stream message ID (format: <ms>-<seq>)."""
    return int(message_id.split("-", maxsplit=1)[0])


def verify_data_streams(
    client: "Redis[bytes]",
    input_streams: list[str],
    output_streams: list[str],
) -> None:
    """Report event counts for input and output streams."""
    total_input = 0
    logger.info("Input streams:")
    for stream in input_streams:
        count = client.xlen(stream)
        logger.info("  {} : {} events", stream, count)
        total_input += count

    total_output = 0
    logger.info("Output streams:")
    for stream in output_streams:
        count = client.xlen(stream)
        logger.info("  {} : {} events", stream, count)
        total_output += count

    if total_input == 0:
        logger.warning("All input streams are empty")
        return

    logger.info("Total: {} input → {} output events", total_input, total_output)

    for stream in output_streams:
        stream_len = client.xlen(stream)
        if stream_len > 0:
            first_entries = client.xrange(stream, count=1)
            last_entries = client.xrevrange(stream, count=1)
            if first_entries and last_entries:
                first_ms = _extract_timestamp_ms(_decode_id(first_entries[0][0]))
                last_ms = _extract_timestamp_ms(_decode_id(last_entries[0][0]))
                duration_s = (last_ms - first_ms) / 1000.0
                if duration_s > 0:
                    throughput = stream_len / duration_s
                    logger.info("  {} throughput: {:.0f} events/sec over {:.2f}s", stream, throughput, duration_s)


def verify_metrics_stream(client: "Redis[bytes]", metrics_stream: str) -> None:
    """Check that metrics stream has events and report sample."""
    metrics_len = client.xlen(metrics_stream)
    logger.info("Metrics stream ({}): {} events", metrics_stream, metrics_len)

    if metrics_len == 0:
        logger.warning("Metrics stream is empty")
        return

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
    verify_data_streams(client, args.input_streams, args.output_streams)

    logger.info("=== Metrics Stream Verification ===")
    verify_metrics_stream(client, args.metrics_stream)

    client.close()


def main() -> None:
    """Entry point for the verifier."""
    args = parse_args()
    run_verify(args)


if __name__ == "__main__":
    main()
