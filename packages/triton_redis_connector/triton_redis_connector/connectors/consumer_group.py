"""Consumer group management for Redis Streams.

Wraps Redis consumer group operations: creation, reading (new and pending),
and acknowledgment. All operations use synchronous redis.Redis client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from loguru import logger
from redis.exceptions import ResponseError

if TYPE_CHECKING:
    from redis import Redis

type _StreamMessages = list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]]


class ConsumerGroupManager:
    """Manages Redis Stream consumer group lifecycle and message reading."""

    def ensure_group_exists(self, client: Redis[bytes], stream: str, group: str) -> None:
        """Create consumer group idempotently, ignoring BUSYGROUP if already exists."""
        try:
            client.xgroup_create(
                name=stream,
                groupname=group,
                id="0",
                mkstream=True,
            )
            logger.info("Created consumer group {} on stream {}", group, stream)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            logger.debug("Consumer group {} already exists on stream {}", group, stream)

    def read_pending_newest_first(
        self,
        client: Redis[bytes],
        stream: str,
        group: str,
        consumer: str,
        count: int,
    ) -> list[tuple[str, dict[bytes, bytes]]]:
        """Read pending messages and return them sorted newest-first.

        Uses XREADGROUP with id "0-0" to fetch unacknowledged messages,
        then reverses the order so newest messages are processed first.

        Returns list of (message_id, fields) tuples.
        """
        raw = cast(
            "_StreamMessages | None",
            client.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={stream: "0-0"},
                count=count,
            ),
        )
        if not raw:
            return []

        entries = self._parse_stream_messages(raw)
        entries.reverse()
        return entries

    def read_new(  # noqa: PLR0913
        self,
        client: Redis[bytes],
        stream: str,
        group: str,
        consumer: str,
        count: int,
        block_ms: int,
    ) -> list[tuple[str, dict[bytes, bytes]]]:
        """Read new (undelivered) messages from the stream via XREADGROUP.

        Uses ">" as the message ID to receive only new messages.
        Blocks for up to block_ms milliseconds waiting for messages.

        Returns list of (message_id, fields) tuples.
        """
        raw = cast(
            "_StreamMessages | None",
            client.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={stream: ">"},
                count=count,
                block=block_ms,
            ),
        )
        if not raw:
            return []

        return self._parse_stream_messages(raw)

    def acknowledge(self, client: Redis[bytes], stream: str, group: str, message_id: str) -> None:
        """Acknowledge a message after successful processing via XACK."""
        client.xack(stream, group, message_id)  # type: ignore[no-untyped-call]

    def _parse_stream_messages(
        self,
        raw: _StreamMessages,
    ) -> list[tuple[str, dict[bytes, bytes]]]:
        entries: list[tuple[str, dict[bytes, bytes]]] = []
        for _stream_name, messages in raw:
            for msg_id_bytes, fields in messages:
                msg_id = msg_id_bytes.decode()
                entries.append((msg_id, fields))
        return entries
