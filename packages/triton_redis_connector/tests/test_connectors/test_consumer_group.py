"""Tests for ConsumerGroupManager with mocked Redis client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from redis.exceptions import ResponseError
from triton_redis_connector.connectors.consumer_group import ConsumerGroupManager


@pytest.fixture
def manager() -> ConsumerGroupManager:
    return ConsumerGroupManager()


@pytest.fixture
def mock_redis() -> MagicMock:
    return MagicMock()


class TestEnsureGroupExists:
    def test_creates_group_with_latest_id(self, manager: ConsumerGroupManager, mock_redis: MagicMock) -> None:
        manager.ensure_group_exists(mock_redis, "stream:input", "workers")
        mock_redis.xgroup_create.assert_called_once_with(
            name="stream:input",
            groupname="workers",
            id="$",
            mkstream=True,
        )

    def test_ignores_busygroup_error(self, manager: ConsumerGroupManager, mock_redis: MagicMock) -> None:
        mock_redis.xgroup_create.side_effect = ResponseError("BUSYGROUP Consumer Group name already exists")
        manager.ensure_group_exists(mock_redis, "stream:input", "workers")

    def test_raises_non_busygroup_error(self, manager: ConsumerGroupManager, mock_redis: MagicMock) -> None:
        mock_redis.xgroup_create.side_effect = ResponseError("WRONGTYPE Operation against a key")
        with pytest.raises(ResponseError, match="WRONGTYPE"):
            manager.ensure_group_exists(mock_redis, "stream:input", "workers")


class TestReadNew:
    def test_returns_empty_when_no_messages(self, manager: ConsumerGroupManager, mock_redis: MagicMock) -> None:
        mock_redis.xreadgroup.return_value = None
        result = manager.read_new(mock_redis, "stream:input", "workers", "c1", 10, 1000)
        assert result == []

    def test_returns_messages_in_order(self, manager: ConsumerGroupManager, mock_redis: MagicMock) -> None:
        mock_redis.xreadgroup.return_value = [
            (
                b"stream:input",
                [
                    (b"1-0", {b"key": b"val1"}),
                    (b"2-0", {b"key": b"val2"}),
                ],
            )
        ]
        result = manager.read_new(mock_redis, "stream:input", "workers", "c1", 10, 1000)
        assert len(result) == 2
        assert result[0][0] == "1-0"
        assert result[1][0] == "2-0"

    def test_uses_gt_id_for_new(self, manager: ConsumerGroupManager, mock_redis: MagicMock) -> None:
        mock_redis.xreadgroup.return_value = None
        manager.read_new(mock_redis, "stream:input", "workers", "c1", 5, 2000)
        mock_redis.xreadgroup.assert_called_once_with(
            groupname="workers",
            consumername="c1",
            streams={"stream:input": ">"},
            count=5,
            block=2000,
        )
