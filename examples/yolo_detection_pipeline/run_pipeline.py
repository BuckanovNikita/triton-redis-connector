"""YOLO detection pipeline: Redis Stream -> Triton -> Redis Stream.

Pipeline graph:
  RedisStreamSource -> EnvelopeExtractor -> ImageDeserializer
    -> PreprocessorNode -> InferenceWorker -> DetectionPostProcessor
    -> OutputSerializer -> RedisStreamSink

Usage:
    uv run python examples/yolo_detection_pipeline/run_pipeline.py

Environment variables (or override in config.yaml):
    REDIS_HOST       (default: redis)
    REDIS_PORT       (default: 6379)
    TRITON_GRPC_URL  (default: triton:8001)
    INPUT_STREAM     (default: test:input_images)
    OUTPUT_STREAM    (default: test:detection_results)
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

from loguru import logger
from redis import Redis
from triton_redis_connector.config.hydra_app import load_config
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

CONFIG_DIR = Path(__file__).resolve().parent
MODEL_NAME = "yolo11s"


def main() -> None:
    """Load config, build pipeline, run."""
    config = load_config(config_name="config", config_dir=CONFIG_DIR)
    logger.info("Starting {}", config.app_name)

    redis_client: Redis[bytes] = Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        decode_responses=False,
    )
    redis_client.ping()
    logger.info("Connected to Redis at {}:{}", config.redis.host, config.redis.port)

    source_cfg = config.sources[0]
    sink_cfg = config.sinks[0]

    group_manager = ConsumerGroupManager()
    input_connector = RedisStreamInputConnector(redis_client, source_cfg, group_manager)
    output_connector = RedisStreamOutputConnector(redis_client, sink_cfg)

    registry = PipelineRegistry()
    registry.register("redis_source", RedisStreamSource(input_connector, source_cfg))
    registry.register("envelope_extractor", EnvelopeExtractor())
    registry.register("image_decoder", ImageDeserializer())
    registry.register("preprocessor", PreprocessorNode(model_name=MODEL_NAME))
    registry.register(
        "inference",
        InferenceWorker(grpc_url=config.inference.grpc_url),
    )
    registry.register("postprocessor", DetectionPostProcessor())
    registry.register("serializer", OutputSerializer())
    registry.register("error_sink", ErrorLogSink())
    registry.register("redis_sink", RedisStreamSink(output_connector))

    app = Application(config=config, registry=registry, queue_bus=QueueBus(config=config.queues))

    def handle_shutdown(signum: int, _frame: types.FrameType | None) -> None:
        logger.info("Received {}, shutting down...", signal.Signals(signum).name)
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
