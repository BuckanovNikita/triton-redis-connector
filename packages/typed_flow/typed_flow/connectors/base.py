"""Internal connector protocols for Redis Stream I/O.

These protocols define the contract that Redis connectors must satisfy.
They are internal to the connector layer -- user-facing code interacts
with the execution variant wrappers (RedisThreadSource, RedisStreamSink).
"""

from typing import Protocol, runtime_checkable

from typed_flow.protocols import Envelope


@runtime_checkable
class InputConnectorProtocol(Protocol):
    """Reads messages from a Redis Stream and wraps them in Envelopes.

    Implementations handle XREADGROUP, consumer group management,
    and deserialization of raw Redis messages into typed payloads.
    """

    def read_new(self, count: int, block_ms: int) -> list[Envelope[bytes]]:
        """Read new messages from the stream via XREADGROUP.

        Returns a list of Envelopes wrapping raw message bytes.
        An empty list means no messages were available within the
        block timeout.
        """
        ...


@runtime_checkable
class OutputConnectorProtocol(Protocol):
    """Writes processed results to a Redis Stream.

    Implementations handle XADD with optional maxlen trimming.
    """

    def write(self, data: dict[str, bytes]) -> str:
        """Write a message to the output stream via XADD.

        Returns the message ID assigned by Redis.
        """
        ...
