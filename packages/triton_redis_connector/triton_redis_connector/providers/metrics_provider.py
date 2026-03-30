"""MetricsProvider — depends on ConfigProvider.

Provides MetricBuffer, MetricEmitter, and collectors at APP scope.
Placeholder for Phase 1 — registrations added in Phase 6A.
"""

from __future__ import annotations

from dishka import Provider

from triton_redis_connector.providers.scopes import ConnectorScope


class MetricsProvider(Provider):
    """Provides metrics collection objects at APP scope.

    Placeholder for Phase 1 — registrations for MetricBuffer,
    MetricEmitter, and collectors will be added in Phase 6A.
    """

    scope = ConnectorScope.APP
