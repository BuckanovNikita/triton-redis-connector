"""Dishka DI providers for the connector pipeline."""

from __future__ import annotations

from triton_redis_connector.providers.config_provider import ConfigProvider
from triton_redis_connector.providers.connector_provider import ConnectorProvider
from triton_redis_connector.providers.engine_provider import EngineProvider
from triton_redis_connector.providers.inference_provider import InferenceProvider
from triton_redis_connector.providers.metrics_provider import MetricsProvider
from triton_redis_connector.providers.node_assembly_provider import NodeAssemblyProvider
from triton_redis_connector.providers.preprocessing_provider import PreprocessingProvider
from triton_redis_connector.providers.scopes import ConnectorScope

__all__ = [
    "ConfigProvider",
    "ConnectorProvider",
    "ConnectorScope",
    "EngineProvider",
    "InferenceProvider",
    "MetricsProvider",
    "NodeAssemblyProvider",
    "PreprocessingProvider",
]
