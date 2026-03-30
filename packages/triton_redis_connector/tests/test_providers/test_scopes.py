"""Tests for custom ConnectorScope hierarchy."""

from __future__ import annotations

from triton_redis_connector.providers.scopes import ConnectorScope


def test_scope_ordering() -> None:
    assert ConnectorScope.APP < ConnectorScope.WORKER
    assert ConnectorScope.WORKER < ConnectorScope.REQUEST
    assert ConnectorScope.APP < ConnectorScope.REQUEST


def test_scope_names() -> None:
    assert ConnectorScope.APP.value.name == "APP"
    assert ConnectorScope.WORKER.value.name == "WORKER"
    assert ConnectorScope.REQUEST.value.name == "REQUEST"


def test_scope_count() -> None:
    assert len(list(ConnectorScope)) == 3


def test_scopes_are_not_skipped() -> None:
    for scope in ConnectorScope:
        assert scope.value.skip is False
