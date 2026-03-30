"""Application entry point with Hydra configuration composition.

Composes the AppConfig from Hydra YAML files, builds a dishka container
with all providers, resolves the Application, and runs the pipeline.
"""

from __future__ import annotations

from typing import Any

from hydra import main as hydra_main  # pyright: ignore[reportMissingTypeStubs,reportUnknownVariableType]
from loguru import logger
from omegaconf import DictConfig, OmegaConf  # pyright: ignore[reportMissingTypeStubs]

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
    from dishka import make_container  # noqa: PLC0415  # pyright: ignore[reportUnknownVariableType]

    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]
    app_config = AppConfig.model_validate(raw)
    logger.info("Starting {} with config: {}", app_config.app_name, app_config)

    container = make_container(  # pyright: ignore[reportUnknownVariableType]
        ConfigProvider(app_config),
        EngineProvider(),
        ConnectorProvider(),
        InferenceProvider(),
        PreprocessingProvider(),
        MetricsProvider(),
        scopes=ConnectorScope,
    )

    try:
        app: Application = container.get(Application)  # pyright: ignore[reportUnknownMemberType]
        app.run()
    finally:
        container.close()  # pyright: ignore[reportUnknownMemberType]


@hydra_main(version_base=None, config_path="conf", config_name="config")  # pyright: ignore[reportUntypedFunctionDecorator,reportArgumentType]
def main(cfg: DictConfig) -> None:
    """Hydra-decorated entry point for the connector pipeline."""
    _run_app(cfg)
