"""Full YOLO detection pipeline: Redis Stream -> Triton Inference -> Redis Stream.

Wires all framework components into a parallel pipeline graph:
  RedisStreamSource (ThreadSource)
    -> ImageDeserializer (ThreadProcessor) — decode JPEG bytes
    -> PreprocessorNode (ThreadProcessor) — letterbox + tensor
    -> InferenceWorker (ProcessPoolProcessor) — Triton gRPC
    -> DetectionPostProcessor (ThreadProcessor) — parse YOLO output
    -> OutputSerializer (ThreadProcessor) — JSON bytes
    -> RedisStreamSink (ThreadSink) — write results

Usage:
    uv run python examples/yolo_detection_pipeline/run_pipeline.py

Environment variables:
    REDIS_HOST       Redis hostname (default: redis)
    REDIS_PORT       Redis port (default: 6379)
    TRITON_GRPC_URL  Triton gRPC endpoint (default: triton:8001)
    MODEL_NAME       YOLO model name (default: yolo11s)
    INPUT_STREAM     Input Redis stream (default: test:input_images)
    OUTPUT_STREAM    Output Redis stream (default: test:detection_results)
"""

from __future__ import annotations

import os
import signal
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

from loguru import logger
from redis import Redis
from triton_redis_connector.config.schema import (
    AppConfig,
    QueueConfig,
    RedisConnectionConfig,
    SinkConfig,
    SourceConfig,
)
from triton_redis_connector.connectors.consumer_group import ConsumerGroupManager
from triton_redis_connector.connectors.redis_stream import (
    RedisStreamInputConnector,
    RedisStreamOutputConnector,
    RedisStreamSink,
    RedisStreamSource,
)
from triton_redis_connector.engine.app import Application
from triton_redis_connector.engine.queue_bus import QueueBus
from triton_redis_connector.engine.registry import PipelineRegistry
from triton_redis_connector.inference.worker import InferenceWorker

from examples.yolo_detection_pipeline.processors import (
    DetectionPostProcessor,
    EnvelopeExtractor,
    ErrorLogSink,
    ImageDeserializer,
    OutputSerializer,
    PreprocessorNode,
)

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
TRITON_GRPC_URL = os.environ.get("TRITON_GRPC_URL", "triton:8001")
MODEL_NAME = os.environ.get("MODEL_NAME", "yolo11s")
INPUT_STREAM = os.environ.get("INPUT_STREAM", "test:input_images")
OUTPUT_STREAM = os.environ.get("OUTPUT_STREAM", "test:detection_results")
CONSUMER_GROUP = "yolo-pipeline"
CONSUMER_NAME = "worker-1"


def build_pipeline() -> Application:
    """Build the full YOLO detection pipeline with all parallel nodes."""
    redis_config = RedisConnectionConfig(host=REDIS_HOST, port=REDIS_PORT)
    source_config = SourceConfig(
        stream_name=INPUT_STREAM,
        consumer_group=CONSUMER_GROUP,
        consumer_name=CONSUMER_NAME,
        event_model="bytes",
        batch_size=1,
        block_ms=2000,
    )
    sink_config = SinkConfig(
        stream_name=OUTPUT_STREAM,
        maxlen=10_000,
    )
    queue_config = QueueConfig(maxsize=100)
    app_config = AppConfig(
        app_name="yolo-detection-pipeline",
        redis=redis_config,
        sources=[source_config],
        sinks=[sink_config],
        queues=queue_config,
    )

    redis_client: Redis[bytes] = Redis(
        host=redis_config.host,
        port=redis_config.port,
        db=redis_config.db,
        decode_responses=False,
    )

    redis_client.ping()
    logger.info("Connected to Redis at {}:{}", redis_config.host, redis_config.port)

    group_manager = ConsumerGroupManager()
    input_connector = RedisStreamInputConnector(redis_client, source_config, group_manager)
    source = RedisStreamSource(input_connector, source_config)

    envelope_extractor = EnvelopeExtractor()
    deserializer = ImageDeserializer()
    preprocessor = PreprocessorNode(model_name=MODEL_NAME, target_size=(640, 640))
    inference_worker = InferenceWorker(grpc_url=TRITON_GRPC_URL, max_concurrent=4)
    postprocessor = DetectionPostProcessor(conf_threshold=0.25, iou_threshold=0.45)
    serializer = OutputSerializer()
    error_sink = ErrorLogSink()

    output_connector = RedisStreamOutputConnector(redis_client, sink_config)
    sink = RedisStreamSink(output_connector)

    registry = PipelineRegistry()
    registry.register("redis_source", source)
    registry.register("envelope_extractor", envelope_extractor)
    registry.register("image_decoder", deserializer)
    registry.register("preprocessor", preprocessor)
    registry.register("inference", inference_worker)
    registry.register("postprocessor", postprocessor)
    registry.register("serializer", serializer)
    registry.register("error_sink", error_sink)
    registry.register("redis_sink", sink)

    queue_bus = QueueBus(config=queue_config)

    return Application(config=app_config, registry=registry, queue_bus=queue_bus)


def main() -> None:
    """Run the YOLO detection pipeline."""
    logger.info(
        "Starting YOLO pipeline: {} -> {} (model={}, triton={})",
        INPUT_STREAM,
        OUTPUT_STREAM,
        MODEL_NAME,
        TRITON_GRPC_URL,
    )

    app = build_pipeline()

    def handle_shutdown(signum: int, _frame: types.FrameType | None) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received {}, shutting down...", sig_name)
        app.shutdown()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

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
