"""Dual-model steel inspection pipeline: Redis Stream -> Branch -> Triton -> Redis Stream.

Pipeline graph:
  RedisStreamSource -> InputExtractor -> CameraBranchRouter
    -> [even] DaliPreprocessor --------+
    -> [odd]  TurboJpegPreprocessor ---+-> MetadataInferenceWorker
                                           -> InspectionPostProcessor
                                               -> OutputSerializer -> RedisStreamSink
                                           -> ErrorLogSink (InferenceError)

Usage:
    uv run python examples/dual_model_pipeline/run_pipeline.py

Environment variables (or override in config.yaml):
    REDIS_HOST       (default: redis)
    REDIS_PORT       (default: 6379)
    TRITON_GRPC_URL  (default: triton:8001)
    INPUT_STREAM     (default: steel:inspection_input)
    OUTPUT_STREAM    (default: steel:inspection_results)
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
from triton_redis_connector.engine.wrappers import DetachedThreadProcessor, DetachedThreadSink

from examples.dual_model_pipeline.processors import (
    CameraBranchRouter,
    DaliPreprocessor,
    ErrorLogSink,
    InputExtractor,
    InspectionPostProcessor,
    MetadataInferenceWorker,
    OutputSerializer,
    TurboJpegPreprocessor,
)

CONFIG_DIR = Path(__file__).resolve().parent


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
    registry.register("input_extractor", DetachedThreadProcessor(InputExtractor()))
    registry.register("branch_router", DetachedThreadProcessor(CameraBranchRouter()))
    registry.register("even_preprocessor", DetachedThreadProcessor(DaliPreprocessor()))
    registry.register("odd_preprocessor", DetachedThreadProcessor(TurboJpegPreprocessor()))
    registry.register(
        "inference",
        MetadataInferenceWorker(grpc_url=config.inference.grpc_url),
    )
    registry.register("postprocessor", DetachedThreadProcessor(InspectionPostProcessor()))
    registry.register("serializer", DetachedThreadProcessor(OutputSerializer()))
    registry.register("error_sink", DetachedThreadSink(ErrorLogSink()))
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
