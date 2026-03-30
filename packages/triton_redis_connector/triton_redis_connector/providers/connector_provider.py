"""ConnectorProvider — depends on ConfigProvider.

Provides Redis connector-related objects at APP scope.
ConsumerGroupManager creation is deferred to Phase 4 when the
provider architecture will support cross-layer wiring.
"""

from __future__ import annotations

from dishka import Provider  # pyright: ignore[reportUnknownVariableType]

from triton_redis_connector.providers.scopes import ConnectorScope


class ConnectorProvider(Provider):
    """Provides Redis connector objects at APP scope.

    ConsumerGroupManager and Redis client providers will be added
    when the application bootstrap wires connectors in Phase 4.
    """

    scope = ConnectorScope.APP
