"""Application entry point with Hydra configuration composition.

Composes the AppConfig from Hydra YAML files, builds a PipelineRegistry
and QueueBus, creates an Application, and runs the pipeline.
"""

from __future__ import annotations

from typing import Any

from hydra import main as hydra_main
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from triton_redis_connector.config.schema import AppConfig
from triton_redis_connector.engine.app import Application
from triton_redis_connector.engine.queue_bus import QueueBus
from triton_redis_connector.engine.registry import PipelineRegistry


def _run_app(cfg: DictConfig) -> None:
    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]
    app_config = AppConfig.model_validate(raw)
    logger.info("Starting {} with config: {}", app_config.app_name, app_config)

    registry = PipelineRegistry()
    queue_bus = QueueBus(config=app_config.queues)
    app = Application(config=app_config, registry=registry, queue_bus=queue_bus)
    app.run()


@hydra_main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra-decorated entry point for the connector pipeline."""
    _run_app(cfg)
