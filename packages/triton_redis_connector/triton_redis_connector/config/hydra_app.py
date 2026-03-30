"""Hydra initialization and configuration composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir  # pyright: ignore[reportMissingTypeStubs]
from loguru import logger
from omegaconf import OmegaConf  # pyright: ignore[reportMissingTypeStubs]

from triton_redis_connector.config.schema import AppConfig

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "conf"


def load_config(
    config_name: str = "config",
    config_dir: Path | None = None,
    overrides: list[str] | None = None,
) -> AppConfig:
    """Load and validate pipeline configuration through Hydra composition.

    Hydra resolves defaults lists, environment variable interpolation
    (``${oc.env:VAR,default}``), and override syntax.  The resulting
    ``OmegaConf`` dict is converted to a plain Python dict and validated
    through the ``AppConfig`` pydantic model.

    Args:
        config_name: Name of the root YAML config (without ``.yaml``).
        config_dir: Absolute path to the Hydra config directory.
            Defaults to the ``conf/`` directory shipped with the package.
        overrides: Hydra CLI-style overrides, e.g.
            ``["redis.host=redis", "inference.grpc_url=triton:8001"]``.

    Returns:
        A fully validated ``AppConfig`` instance.

    """
    resolved_dir = str(config_dir or _DEFAULT_CONFIG_DIR)
    resolved_overrides = overrides or []

    with initialize_config_dir(
        config_dir=resolved_dir,
        version_base="1.3",
    ):
        cfg = compose(config_name=config_name, overrides=resolved_overrides)

    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[assignment]
    logger.debug("Loaded raw config from Hydra: {}", raw)
    return AppConfig.model_validate(raw)
