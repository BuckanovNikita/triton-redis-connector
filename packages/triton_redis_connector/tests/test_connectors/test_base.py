"""Tests for connector protocol isinstance checks."""

from __future__ import annotations

from triton_redis_connector.connectors.base import (
    InputConnectorProtocol,
    OutputConnectorProtocol,
)
from triton_redis_connector.protocols import Envelope


class DummyInputConnector:
    def read_new(self, count: int, block_ms: int) -> list[Envelope[bytes]]:
        return [Envelope(payload=b"test", source_stream="s", message_id="1-0")]

    def read_pending(self, count: int) -> list[Envelope[bytes]]:
        return []

    def acknowledge(self, message_id: str) -> None:
        pass


class DummyOutputConnector:
    def write(self, data: dict[str, bytes]) -> str:
        return "1-0"


class NotAConnector:
    def unrelated(self) -> None:
        pass


def test_input_connector_protocol_isinstance() -> None:
    connector = DummyInputConnector()
    assert isinstance(connector, InputConnectorProtocol)


def test_output_connector_protocol_isinstance() -> None:
    connector = DummyOutputConnector()
    assert isinstance(connector, OutputConnectorProtocol)


def test_non_connector_fails_isinstance() -> None:
    obj = NotAConnector()
    assert not isinstance(obj, InputConnectorProtocol)
    assert not isinstance(obj, OutputConnectorProtocol)


def test_input_connector_read_new() -> None:
    connector = DummyInputConnector()
    results = connector.read_new(count=10, block_ms=1000)
    assert len(results) == 1
    assert results[0].payload == b"test"
    assert results[0].source_stream == "s"


def test_input_connector_read_pending() -> None:
    connector = DummyInputConnector()
    results = connector.read_pending(count=10)
    assert results == []


def test_output_connector_write() -> None:
    connector = DummyOutputConnector()
    msg_id = connector.write({b"key": b"value"})
    assert msg_id == "1-0"
