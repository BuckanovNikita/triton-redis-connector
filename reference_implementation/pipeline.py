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
import time
from typing import TYPE_CHECKING

from loguru import logger
from redis import Redis
from typed_flow.config.schema import (
    AppConfig,
    RedisConnectionConfig,
    SinkConfig,
    SourceConfig,
)
from typed_flow.connectors.consumer_group import ConsumerGroupManager
from typed_flow.connectors.redis_stream import (
    RedisStreamInputConnector,
    RedisStreamOutputConnector,
)
from typed_flow.engine.app import Application
from typed_flow.engine.events import FrameworkMeta, PipelineEvent, Unprocessed
from typed_flow.errors.models import ConnectorError
from typed_flow.protocols import BaseThreadProcessor, BaseThreadSink, BaseThreadSource

from reference_implementation.models import PassthroughResult, SampleEvent

if TYPE_CHECKING:
    import types

INPUT_STREAM = "restreamer:input"
OUTPUT_STREAM = "restreamer:output"
CONSUMER_GROUP = "restreamer-group"
CONSUMER_NAME = "restreamer-0"

_SourceEvent = PipelineEvent[SampleEvent, Unprocessed]
_SinkEvent = PipelineEvent[SampleEvent, PassthroughResult]


class RestreamerSource(BaseThreadSource[_SourceEvent | ConnectorError]):
    """Reads SampleEvent from a Redis stream via consumer group."""

    def __init__(
        self,
        connector: RedisStreamInputConnector,
        config: SourceConfig,
    ) -> None:
        """Initialize with Redis connector and source config."""
        super().__init__()
        self._connector = connector
        self._config = config
        self.name = "restreamer_source"

    def __call__(self) -> _SourceEvent | ConnectorError | None:  # noqa: D102
        envelopes = self._connector.read_new(
            count=self._config.batch_size,
            block_ms=self._config.block_ms,
        )
        if not envelopes:
            return None
        envelope = envelopes[0]
        fields = dict(envelope.original_fields)
        if envelope.payload:
            fields["data"] = envelope.payload.decode()
        try:
            event = SampleEvent.model_validate(fields)
        except Exception as exc:  # noqa: BLE001
            return ConnectorError(
                connector_name=self.name,
                stream_name=envelope.source_stream,
                message_id=envelope.message_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
                timestamp_ms=int(time.time() * 1000),
            )
        fw = FrameworkMeta(
            source_stream=envelope.source_stream,
            message_id=envelope.message_id,
        )
        return _SourceEvent(fw=fw, context=event, payload=Unprocessed())


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


class RestreamerSink(BaseThreadSink[_SinkEvent]):
    """Writes processed events to the output Redis stream."""

    def __init__(self, connector: RedisStreamOutputConnector) -> None:
        """Initialize with output connector."""
        super().__init__()
        self._output = connector
        self.name = "restreamer_sink"

    def __call__(self, event: _SinkEvent) -> None:  # noqa: D102
        ctx = event.context
        result = event.payload
        data: dict[str, bytes] = {
            "sequence_id": str(ctx.sequence_id).encode(),
            "timestamp": str(ctx.timestamp).encode(),
            "payload": ctx.payload.encode(),
            "result": result.model_dump_json().encode(),
        }
        self._output.write(data)


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

    redis_client: Redis[bytes] = Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=0,
        decode_responses=False,
    )
    redis_client.ping()
    logger.info("Redis connected at {}:{}", args.redis_host, args.redis_port)

    source_cfg = SourceConfig(
        stream_name=args.input_stream,
        consumer_group=CONSUMER_GROUP,
        consumer_name=CONSUMER_NAME,
        event_model="SampleEvent",
        batch_size=1,
        block_ms=args.block_ms,
    )
    sink_cfg = SinkConfig(
        stream_name=args.output_stream,
        maxlen=100_000,
        approximate_maxlen=True,
    )
    app_config = AppConfig(
        redis=RedisConnectionConfig(
            host=args.redis_host,
            port=args.redis_port,
        ),
    )

    group_manager = ConsumerGroupManager()
    input_connector = RedisStreamInputConnector(redis_client, source_cfg, group_manager)
    output_connector = RedisStreamOutputConnector(redis_client, sink_cfg)

    source = RestreamerSource(connector=input_connector, config=source_cfg)
    processor = PassthroughProcessor()
    sink = RestreamerSink(connector=output_connector)
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
