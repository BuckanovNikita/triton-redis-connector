"""Application entry point with Hydra configuration composition.

Composes the AppConfig from Hydra YAML files, builds a dishka container
with all providers, resolves the Application, and runs the pipeline.
"""

from __future__ import annotations

from typing import Any

from dishka import make_container
from hydra import main as hydra_main
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from triton_redis_connector.config.schema import AppConfig
from triton_redis_connector.engine.app import Application
from triton_redis_connector.providers import (
    ConfigProvider,
    ConnectorProvider,
    ConnectorScope,
    EngineProvider,
    InferenceProvider,
    MetricsProvider,
    PreprocessingProvider,
)


def _run_app(cfg: DictConfig) -> None:
    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]
    app_config = AppConfig.model_validate(raw)
    logger.info("Starting {} with config: {}", app_config.app_name, app_config)

    container = make_container(
        ConfigProvider(app_config),
        EngineProvider(),
        ConnectorProvider(),
        InferenceProvider(),
        PreprocessingProvider(),
        MetricsProvider(),
        scopes=ConnectorScope,
    )

    try:
        app: Application = container.get(Application)
        app.run()
    finally:
        container.close()


@hydra_main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra-decorated entry point for the connector pipeline."""
    _run_app(cfg)
