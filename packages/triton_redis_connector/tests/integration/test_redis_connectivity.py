"""Integration tests for Redis connectivity and stream operations."""

from __future__ import annotations

import pytest
import redis.asyncio

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group(name="integration")]


class TestRedisConnectivity:
    async def test_redis_ping(self, redis_client: redis.asyncio.Redis[str]) -> None:
        result = await redis_client.ping()
        assert result is True

    async def test_xadd_and_xlen(self, redis_client: redis.asyncio.Redis[str]) -> None:
        stream = "test:stream:basic"
        await redis_client.xadd(stream, {"key": "value1"})
        assert await redis_client.xlen(stream) == 1
        await redis_client.xadd(stream, {"key": "value2"})
        assert await redis_client.xlen(stream) == 2

    async def test_xadd_and_xread(self, redis_client: redis.asyncio.Redis[str]) -> None:
        stream = "test:stream:read"
        await redis_client.xadd(stream, {"field_a": "hello", "field_b": "world"})

        result = await redis_client.xread({stream: "0-0"}, count=1)
        assert len(result) == 1
        messages = result[0][1]
        assert len(messages) == 1
        _, fields = messages[0]
        assert fields["field_a"] == "hello"
        assert fields["field_b"] == "world"

    async def test_xgroup_create_and_xreadgroup(
        self,
        redis_client: redis.asyncio.Redis[str],
    ) -> None:
        stream = "test:stream:group"
        group = "test_group"
        consumer = "test_consumer"

        await redis_client.xadd(stream, {"data": "payload"})
        await redis_client.xgroup_create(stream, group, id="0")

        result = await redis_client.xreadgroup(group, consumer, {stream: ">"}, count=1)
        assert len(result) == 1
        messages = result[0][1]
        assert len(messages) == 1
        _, fields = messages[0]
        assert fields["data"] == "payload"

    async def test_xack_removes_from_pending(
        self,
        redis_client: redis.asyncio.Redis[str],
    ) -> None:
        stream = "test:stream:ack"
        group = "test_group"
        consumer = "test_consumer"

        await redis_client.xadd(stream, {"data": "to_ack"})
        await redis_client.xgroup_create(stream, group, id="0")

        result = await redis_client.xreadgroup(group, consumer, {stream: ">"}, count=1)
        message_id = result[0][1][0][0]

        await redis_client.xack(stream, group, message_id)
        pending = await redis_client.xpending(stream, group)
        assert pending["pending"] == 0

    async def test_xadd_maxlen_trims(self, redis_client: redis.asyncio.Redis[str]) -> None:
        stream = "test:stream:maxlen"
        for i in range(5):
            await redis_client.xadd(stream, {"i": str(i)}, maxlen=2)

        length = await redis_client.xlen(stream)
        assert length <= 2
