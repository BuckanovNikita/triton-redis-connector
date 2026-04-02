"""Hydra configuration loading via @hydra.main decorator only.

This module provides ``load_config_from_dict_config`` for use inside a
``@hydra.main`` decorated function. The compose API is intentionally
not used -- all config resolution goes through Hydra's standard entry
point.

Usage::

    import hydra
    from omegaconf import DictConfig
    from triton_redis_connector.config.hydra_app import (
        load_config_from_dict_config,
        get_default_config_dir,
    )


    @hydra.main(
        version_base="1.3",
        config_path=get_default_config_dir(),
        config_name="config",
    )
    def main(cfg: DictConfig) -> None:
        app_config = load_config_from_dict_config(cfg)
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from triton_redis_connector.config.schema import AppConfig

_DEFAULT_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "conf")


def load_config_from_dict_config(cfg: DictConfig) -> AppConfig:
    """Convert a Hydra DictConfig into a validated AppConfig.

    Intended for use inside a ``@hydra.main`` decorated function where
    Hydra has already resolved defaults, environment variable interpolation,
    and CLI overrides.

    Args:
        cfg: The OmegaConf DictConfig produced by Hydra.

    Returns:
        A fully validated ``AppConfig`` instance.

    """
    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]
    logger.debug("Loaded raw config from Hydra: {}", raw)
    return AppConfig.model_validate(raw)


def get_default_config_dir() -> str:
    """Return the absolute path to the shipped conf/ directory."""
    return _DEFAULT_CONFIG_DIR
