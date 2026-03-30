"""Custom scope hierarchy for the connector pipeline.

APP scope: application-lifetime singletons (config, registry, redis pool).
WORKER scope: per-process singletons for ProcessPoolProcessor workers.
REQUEST scope: per-event-dispatch context in the main loop.
"""

from __future__ import annotations

from dishka import BaseScope, new_scope


class ConnectorScope(BaseScope):
    """Three-level scope hierarchy for the connector pipeline."""

    APP = new_scope("APP")
    WORKER = new_scope("WORKER")
    REQUEST = new_scope("REQUEST")
