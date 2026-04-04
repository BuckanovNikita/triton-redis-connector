"""ProcessorFactory protocol for shared-state processor creation.

Framework calls create_shared() once, then create_processor(shared)
N times (one per worker).
"""

from typing import Protocol, TypeVar, runtime_checkable

TShared = TypeVar("TShared")
TProcessor_co = TypeVar("TProcessor_co", covariant=True)


@runtime_checkable
class ProcessorFactory(Protocol[TShared, TProcessor_co]):
    """Factory for creating processor instances with shared state.

    Framework calls create_shared() once, then create_processor(shared)
    N times (one per worker). User is responsible for thread/process
    safety of the shared context.
    """

    def create_shared(self) -> TShared:
        """Create the shared context object (called once)."""
        ...

    def create_processor(self, shared: TShared) -> TProcessor_co:
        """Create a processor instance with access to shared context."""
        ...
