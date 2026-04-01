"""Application factory — builds a fully wired Application from config.

Creates the Redis client, stream connectors, source/sink nodes, registers
shipped processors from config fields, and returns a ready-to-run Application.

Shipped processors resolved from config:
- ``preprocessing.decoder`` + ``pipeline.model_name`` → preprocessor
- ``inference.grpc_url`` → InferenceWorker
- EventSerializer and LogErrorSink are always registered.

Passed from code (not config):
- ``event_model`` → EnvelopeExtractor
- ``postprocessor`` → any callable (e.g., YoloPostProcessorBridge)
- ``extra_nodes`` → user-defined processors
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from redis import Redis

from triton_redis_connector.config.schema import DecoderType, NodeVariant
from triton_redis_connector.connectors.consumer_group import ConsumerGroupManager
from triton_redis_connector.connectors.redis_stream import (
    RedisStreamInputConnector,
    RedisStreamOutputConnector,
    RedisStreamSink,
    RedisStreamSource,
)
from triton_redis_connector.engine.app import Application
from triton_redis_connector.engine.assembly import assemble_all, wrap_with_variant
from triton_redis_connector.engine.queue_bus import QueueBus
from triton_redis_connector.engine.registry import PipelineRegistry
from triton_redis_connector.engine.shipped_processors import (
    DaliInferenceInputBuilder,
    EnvelopeExtractor,
    EventSerializer,
    LogErrorSink,
)
from triton_redis_connector.inference.worker import InferenceWorker
from triton_redis_connector.providers.shipped_factories import ConfigurableClientPreprocessor

if TYPE_CHECKING:
    from pydantic import BaseModel

    from triton_redis_connector.config.schema import AppConfig
    from triton_redis_connector.protocols import LifecycleService, MetricRecorder


def _register_shipped(
    registry: PipelineRegistry,
    name: str,
    instance: object,
    variant: NodeVariant,
) -> None:
    wrapped = wrap_with_variant(instance, variant)
    registry.register(name, wrapped)


def build_application(  # noqa: PLR0913
    config: AppConfig,
    *,
    event_model: type[BaseModel] | None = None,
    postprocessor: object | None = None,
    extra_nodes: list[tuple[str, object]] | None = None,
    metric_buffer: MetricRecorder | None = None,
    triton_collector: LifecycleService | None = None,
    metric_emitter: LifecycleService | None = None,
) -> Application:
    """Build a fully wired Application from an AppConfig."""
    redis_client: Redis[bytes] = Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        decode_responses=False,
    )
    redis_client.ping()
    logger.info("Connected to Redis at {}:{}", config.redis.host, config.redis.port)

    group_manager = ConsumerGroupManager()
    registry = PipelineRegistry()

    _register_redis_connectors(registry, redis_client, config, group_manager)
    _register_shipped_processors(registry, config, event_model=event_model, postprocessor=postprocessor)

    if config.nodes:
        assemble_all(config.nodes, registry, metric_recorder=metric_buffer)

    if extra_nodes:
        for node_name, node in extra_nodes:
            registry.register(node_name, node)

    return Application(
        config=config,
        registry=registry,
        queue_bus=QueueBus(config=config.queues),
        metric_buffer=metric_buffer,
        triton_collector=triton_collector,
        metric_emitter=metric_emitter,
    )


def _register_redis_connectors(
    registry: PipelineRegistry,
    redis_client: Redis[bytes],
    config: AppConfig,
    group_manager: ConsumerGroupManager,
) -> None:
    multi_source = len(config.sources) > 1
    for i, source_cfg in enumerate(config.sources):
        input_connector = RedisStreamInputConnector(redis_client, source_cfg, group_manager)
        source = RedisStreamSource(input_connector, source_cfg)
        name = f"redis_source_{i}" if multi_source else "redis_source"
        registry.register(name, source)

    multi_sink = len(config.sinks) > 1
    for i, sink_cfg in enumerate(config.sinks):
        output_connector = RedisStreamOutputConnector(redis_client, sink_cfg)
        sink = RedisStreamSink(output_connector)
        name = f"redis_sink_{i}" if multi_sink else "redis_sink"
        registry.register(name, sink)


def _register_shipped_processors(
    registry: PipelineRegistry,
    config: AppConfig,
    *,
    event_model: type[BaseModel] | None = None,
    postprocessor: object | None = None,
) -> None:
    pipeline = config.pipeline

    if event_model is not None:
        _register_shipped(registry, "extractor", EnvelopeExtractor(event_model), NodeVariant.THREAD_PROCESSOR)

    if pipeline.model_name:
        preprocessor = _build_preprocessor(config)
        _register_shipped(registry, "preprocessor", preprocessor, NodeVariant.THREAD_PROCESSOR)

    _register_shipped(
        registry,
        "inference",
        InferenceWorker(grpc_url=config.inference.grpc_url),
        NodeVariant.THREAD_PROCESSOR,
    )

    if postprocessor is not None:
        _register_shipped(registry, "postprocessor", postprocessor, NodeVariant.THREAD_PROCESSOR)

    _register_shipped(registry, "serializer", EventSerializer(), NodeVariant.THREAD_PROCESSOR)
    _register_shipped(registry, "error_sink", LogErrorSink(), NodeVariant.THREAD_SINK)


def _build_preprocessor(config: AppConfig) -> DaliInferenceInputBuilder | ConfigurableClientPreprocessor:
    pipeline = config.pipeline
    output_names = tuple(pipeline.model_output_names)

    if config.preprocessing.decoder == DecoderType.DALI:
        return DaliInferenceInputBuilder(
            model_name=pipeline.model_name,
            input_name=pipeline.model_input_name,
            output_names=output_names,
        )

    return ConfigurableClientPreprocessor(
        model_name=pipeline.model_name,
        decoder_type=config.preprocessing.decoder.value,
        target_size=config.preprocessing.default_target_size,
        input_name=pipeline.model_input_name,
        output_names=output_names,
    )
