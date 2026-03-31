"""Broadcast router implementation."""

from __future__ import annotations

from triton_redis_connector.routers.protocol import RoutingTarget


class BroadcastRouter:
    """Routes every event to all configured models."""

    def __init__(self, model_names: list[str]) -> None:  # noqa: D107
        self._targets = [RoutingTarget(model_name=name) for name in model_names]

    def route(self, event: object) -> list[RoutingTarget]:  # noqa: ARG002
        """Return routing targets for all configured models."""
        return list(self._targets)
