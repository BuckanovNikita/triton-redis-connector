"""Tests for EventContext and contextvars-based context management."""

from __future__ import annotations

import threading

import pytest
from triton_redis_connector.engine.context import (
    EventContext,
    _context_var,
    current_context,
    set_context,
)


class TestEventContext:
    def test_is_frozen(self) -> None:
        ctx = EventContext(message_id="1-0", source_stream="stream:cam", created_at=100.0)
        with pytest.raises(AttributeError):
            ctx.message_id = "2-0"  # type: ignore[misc]

    def test_stores_all_fields(self) -> None:
        ctx = EventContext(message_id="msg-1", source_stream="stream:input", created_at=42.5)
        assert ctx.message_id == "msg-1"
        assert ctx.source_stream == "stream:input"
        assert ctx.created_at == 42.5


class TestCurrentContext:
    def test_raises_lookup_error_when_unset(self) -> None:
        token = _context_var.set(EventContext(message_id="", source_stream="", created_at=0.0))
        _context_var.reset(token)
        with pytest.raises(LookupError):
            current_context()

    def test_returns_set_context(self) -> None:
        ctx = EventContext(message_id="abc", source_stream="stream:test", created_at=1.0)
        token = set_context(ctx)
        try:
            result = current_context()
            assert result is ctx
            assert result.message_id == "abc"
        finally:
            _context_var.reset(token)

    def test_set_context_returns_token(self) -> None:
        ctx = EventContext(message_id="x", source_stream="s", created_at=0.0)
        token = set_context(ctx)
        _context_var.reset(token)

        with pytest.raises(LookupError):
            current_context()

    def test_context_is_thread_local(self) -> None:
        ctx = EventContext(message_id="main", source_stream="main-stream", created_at=1.0)
        token = set_context(ctx)

        seen_in_thread: list[bool] = []

        def thread_fn() -> None:
            try:
                current_context()
                seen_in_thread.append(True)
            except LookupError:
                seen_in_thread.append(False)

        t = threading.Thread(target=thread_fn)
        t.start()
        t.join()

        _context_var.reset(token)

        assert seen_in_thread == [False]

    def test_nested_context_restore(self) -> None:
        outer = EventContext(message_id="outer", source_stream="s", created_at=1.0)
        inner = EventContext(message_id="inner", source_stream="s", created_at=2.0)

        outer_token = set_context(outer)
        assert current_context().message_id == "outer"

        inner_token = set_context(inner)
        assert current_context().message_id == "inner"

        _context_var.reset(inner_token)
        assert current_context().message_id == "outer"

        _context_var.reset(outer_token)
