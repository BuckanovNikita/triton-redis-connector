"""Stream generator protocol for producing typed events into Redis streams."""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class StreamGenerator(Protocol):
    """Protocol for generators that produce typed events for a specific Redis stream.

    Implementors must provide:
    - ``stream_name``: default target stream name (can be overridden by the app)
    - ``event_type``: the BaseModel subclass this generator produces
    - ``__call__``: returns an iterator of BaseModel instances
    """

    stream_name: str
    event_type: type[BaseModel]

    def __call__(self) -> Iterator[BaseModel]:
        """Return an iterator of typed events."""
        ...
