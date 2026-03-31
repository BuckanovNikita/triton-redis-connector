"""Optional concrete base classes for parallel execution variant protocols.

These base classes eliminate boilerplate (stop events) while satisfying
the protocol contracts structurally. Users define only business-related
__init__ and __call__ logic.

Usage is OPTIONAL — users can always implement the Protocol directly.
Only parallel variants are provided (inline variants have no boilerplate).
"""

from __future__ import annotations

import threading
from abc import abstractmethod
from typing import Any


class BaseThreadSource[TOut]:
    """Base for ThreadSource. Provides stop event; subclass defines __call__."""

    def __init__(self) -> None:  # noqa: D107
        self.stop = threading.Event()

    @abstractmethod
    def __call__(self) -> TOut | None:
        """Produce an event or return None if nothing available."""
        ...


class BaseProcessSource[TOut]:
    """Base for ProcessSource. Provides stop event; subclass defines __call__."""

    def __init__(self) -> None:  # noqa: D107
        self.stop = threading.Event()

    @abstractmethod
    def __call__(self) -> TOut | None:
        """Produce an event or return None if nothing available."""
        ...


class BaseThreadProcessor[TIn, TOut]:
    """Base for ThreadProcessor. Provides stop event; subclass defines __call__."""

    def __init__(self) -> None:  # noqa: D107
        self.stop = threading.Event()

    @abstractmethod
    def __call__(self, event: TIn) -> TOut | None:
        """Transform an event or return None to filter it out."""
        ...


class BaseThreadPoolProcessor[TIn, TOut]:
    """Base for ThreadPoolProcessor. Provides stop event; subclass defines __call__."""

    def __init__(self) -> None:  # noqa: D107
        self.stop = threading.Event()

    @abstractmethod
    def __call__(self, event: TIn) -> TOut | None:
        """Transform an event or return None to filter it out."""
        ...


class BaseProcessPoolProcessor[TIn, TOut]:
    """Base for ProcessPoolProcessor. Provides stop event; subclass defines __call__."""

    def __init__(self) -> None:  # noqa: D107
        self.stop = threading.Event()

    @abstractmethod
    def __call__(self, event: TIn) -> TOut | None:
        """Transform an event or return None to filter it out."""
        ...


class BaseThreadSink[TIn]:
    """Base for ThreadSink. Provides stop event; subclass defines __call__."""

    def __init__(self) -> None:  # noqa: D107
        self.stop = threading.Event()

    @abstractmethod
    def __call__(self, event: TIn) -> Any | None:  # noqa: ANN401
        """Consume an event and optionally return a metric event."""
        ...
