"""InferenceProvider — depends on ConfigProvider.

Provides inference worker configuration at APP scope. The actual
tritonclient instances are created per-worker at WORKER scope
(via a separate WorkerProvider inside each process).
"""

from __future__ import annotations

from dishka import Provider

from triton_redis_connector.providers.scopes import ConnectorScope


class InferenceProvider(Provider):
    """Provides inference-related config at APP scope.

    Placeholder for Phase 1 — WORKER-scope registrations for
    per-process tritonclient and ModelVersionTracker will be added
    in Phase 4A when the ProcessPoolProcessor is implemented.
    """

    scope = ConnectorScope.APP
