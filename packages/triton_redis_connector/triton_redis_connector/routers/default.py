"""Field-based router implementation."""

from __future__ import annotations

from triton_redis_connector.routers.protocol import RoutingTarget


class FieldBasedRouter:
    """Routes based on a field in the input event. Falls back to default_model."""

    def __init__(self, default_model: str, model_field: str = "model_name") -> None:  # noqa: D107
        self._default_model = default_model
        self._model_field = model_field

    def route(self, event: object) -> list[RoutingTarget]:
        """Return a single routing target derived from the event's model field."""
        model_name = getattr(event, self._model_field, None)
        if not isinstance(model_name, str) or not model_name:
            model_name = self._default_model
        if not model_name:
            return []
        return [RoutingTarget(model_name=model_name)]
