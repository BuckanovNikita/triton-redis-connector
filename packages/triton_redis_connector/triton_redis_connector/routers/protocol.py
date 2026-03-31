"""Router protocol and routing target model."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict

TIn_contra = TypeVar("TIn_contra", contravariant=True)


class RoutingTarget(BaseModel):
    """Describes a model endpoint to route an event to."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    model_version: str = ""


@runtime_checkable
class RouterProtocol(Protocol[TIn_contra]):
    """Routes an input event to one or more model targets."""

    def route(self, event: TIn_contra) -> list[RoutingTarget]:
        """Return the list of routing targets for the given event."""
        ...
