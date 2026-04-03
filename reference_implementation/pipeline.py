"""Restreamer pipeline: reads events from one Redis stream, writes to another.

Pure typed-flow benchmark — no Triton, no ML. Measures framework throughput
and correctness (zero event loss, zero errors).

Usage:
    PYTHONPATH=. uv run python reference_implementation/pipeline.py
    PYTHONPATH=. uv run python reference_implementation/pipeline.py --redis-host redis
"""

from __future__ import annotations

import argparse
import signal
import sys
from typing import TYPE_CHECKING

from loguru import logger
from typed_flow.config.schema import (
    AppConfig,
    RedisConnectionConfig,
    SinkConfig,
    SourceConfig,
)
from typed_flow.connectors import RedisThreadSink, RedisThreadSource
from typed_flow.engine.app import Application
from typed_flow.engine.events import PipelineEvent, Unprocessed
from typed_flow.errors.models import ConnectorError
from typed_flow.protocols import BaseThreadProcessor, BaseThreadSink

from reference_implementation.models import PassthroughResult, SampleEvent

if TYPE_CHECKING:
    import types

INPUT_STREAM = "restreamer:input"
OUTPUT_STREAM = "restreamer:output"
CONSUMER_GROUP = "restreamer-group"

_SourceEvent = PipelineEvent[SampleEvent, Unprocessed]
_SinkEvent = PipelineEvent[SampleEvent, PassthroughResult]


class RestreamerSource(RedisThreadSource[SampleEvent]):
    """Reads SampleEvent from a Redis stream via consumer group."""


class RestreamerSink(RedisThreadSink[SampleEvent, PassthroughResult]):
    """Writes processed events to the output Redis stream."""


class PassthroughProcessor(BaseThreadProcessor[_SourceEvent, _SinkEvent]):
    """Converts Unprocessed events into PassthroughResult for the sink."""

    def __call__(self, event: _SourceEvent) -> _SinkEvent | None:  # noqa: D102
        ctx = event.context
        result = PassthroughResult(
            sequence_id=ctx.sequence_id,
            timestamp=ctx.timestamp,
            payload=ctx.payload,
        )
        return _SinkEvent(fw=event.fw, context=event.context, payload=result)


class ErrorSink(BaseThreadSink[ConnectorError]):
    """Logs deserialization errors from the source connector."""

    def __call__(self, event: ConnectorError) -> None:  # noqa: D102
        logger.error(
            "Connector error on {}: {}",
            event.stream_name,
            event.error_message,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the restreamer pipeline."""
    parser = argparse.ArgumentParser(description="Restreamer benchmark pipeline")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--input-stream", default=INPUT_STREAM)
    parser.add_argument("--output-stream", default=OUTPUT_STREAM)
    parser.add_argument("--block-ms", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    """Build and run the restreamer pipeline."""
    args = parse_args()

    redis_cfg = RedisConnectionConfig(host=args.redis_host, port=args.redis_port)

    source_cfg = SourceConfig(
        stream_name=args.input_stream,
        consumer_group=CONSUMER_GROUP,
        event_model="SampleEvent",
        batch_size=1,
        block_ms=args.block_ms,
    )
    sink_cfg = SinkConfig(
        stream_name=args.output_stream,
        maxlen=100_000,
        approximate_maxlen=True,
    )
    app_config = AppConfig(redis=redis_cfg)

    source = RestreamerSource(redis=redis_cfg, source=source_cfg, name="restreamer_source")
    processor = PassthroughProcessor()
    sink = RestreamerSink(redis=redis_cfg, sink=sink_cfg, name="restreamer_sink")
    error_sink = ErrorSink()

    app = Application.build(
        app_config,
        processors=[source, processor, sink, error_sink],
    )

    def handle_shutdown(signum: int, _frame: types.FrameType | None) -> None:
        logger.info("Received {}, shutting down...", signal.Signals(signum).name)
        app.shutdown()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("Restreamer pipeline: {} -> {}", args.input_stream, args.output_stream)

    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, shutting down...")
        app.shutdown()
    except (RuntimeError, ValueError, OSError):
        logger.exception("Pipeline crashed")
        app.shutdown()
        sys.exit(1)

    logger.info("Pipeline stopped gracefully")


if __name__ == "__main__":
    main()
