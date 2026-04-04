"""Application entry point with Hydra configuration composition.

Composes the TritonAppConfig from Hydra YAML files, builds a PipelineRegistry
and QueueBus, creates an Application, and runs the pipeline.
"""

import os
from typing import Any

import redis as redis_lib
import tritonclient.grpc as grpcclient
from hydra import main as hydra_main
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from typed_flow.engine.app import Application
from typed_flow.engine.queue_bus import QueueBus
from typed_flow.engine.registry import PipelineRegistry
from typed_flow.metrics.buffer import MetricBuffer
from typed_flow.metrics.emitter import MetricEmitter

from triton_redis_connector.config.schema import TritonAppConfig
from triton_redis_connector.metrics.triton_collector import TritonMetricsCollector

APP_NAME = os.environ.get("APP_NAME", "triton-connector")


def _run_app(cfg: DictConfig) -> None:
    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]
    app_config = TritonAppConfig.model_validate(raw)
    logger.info("Starting {} with config: {}", APP_NAME, app_config)

    registry = PipelineRegistry()
    queue_bus = QueueBus(config=app_config.queues)

    metric_buffer = MetricBuffer()

    triton_client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(
        url=app_config.inference.grpc_url,
    )
    triton_collector = TritonMetricsCollector(
        client=triton_client,
        buffer=metric_buffer,
        poll_interval_s=app_config.inference.metrics_poll_interval_s,
    )

    metric_emitter: MetricEmitter | None = None
    if app_config.metrics.output_stream:
        redis_client: redis_lib.Redis[bytes] = redis_lib.Redis(
            host=app_config.redis.host,
            port=app_config.redis.port,
            db=app_config.redis.db,
            decode_responses=False,
        )
        metric_emitter = MetricEmitter(
            buffer=metric_buffer,
            client=redis_client,
            stream=app_config.metrics.output_stream,
            interval_s=app_config.metrics.emit_interval_s,
            known_metrics=registry.all_counter_names,
            app_name=APP_NAME,
        )

    app = Application(
        config=app_config,
        registry=registry,
        queue_bus=queue_bus,
        metric_buffer=metric_buffer,
        extra_service=triton_collector,
        metric_emitter=metric_emitter,
    )
    app.run()


@hydra_main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra-decorated entry point for the connector pipeline."""
    _run_app(cfg)
