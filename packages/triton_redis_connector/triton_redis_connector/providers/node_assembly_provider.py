"""NodeAssemblyProvider — reads AppConfig.nodes and assembles pipeline nodes.

Depends on ConfigProvider (AppConfig) and EngineProvider (PipelineRegistry).
Runs assembly before Application starts via dishka dependency ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

from dishka import Provider, provide

from triton_redis_connector.config.schema import AppConfig  # noqa: TC001 — dishka needs runtime access
from triton_redis_connector.engine.assembly import assemble_all
from triton_redis_connector.engine.registry import PipelineRegistry  # noqa: TC001
from triton_redis_connector.providers.scopes import ConnectorScope


@dataclass(frozen=True)
class NodesAssembled:
    """Sentinel indicating config-driven nodes have been registered."""

    count: int


class NodeAssemblyProvider(Provider):
    """Assembles config-driven nodes and registers them in the pipeline."""

    scope = ConnectorScope.APP

    @provide
    def assemble_nodes(self, config: AppConfig, registry: PipelineRegistry) -> NodesAssembled:  # noqa: D102
        assemble_all(config.nodes, registry)
        return NodesAssembled(count=len(config.nodes))
