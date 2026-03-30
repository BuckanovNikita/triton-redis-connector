"""PreprocessingProvider — depends on ConfigProvider.

Provides image decoder instances at APP scope.
Placeholder for Phase 1 — registrations added in Phase 4A.
"""

from __future__ import annotations

from dishka import Provider

from triton_redis_connector.providers.scopes import ConnectorScope


class PreprocessingProvider(Provider):
    """Provides preprocessing objects at APP scope.

    Placeholder for Phase 1 — registrations for ImageDecoder
    instances will be added in Phase 4A.
    """

    scope = ConnectorScope.APP
