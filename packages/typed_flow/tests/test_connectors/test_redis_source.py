"""Tests for Redis source __init_subclass__ type extraction."""

import pytest
from pydantic import BaseModel
from typed_flow.connectors.redis_source import RedisProcessSource, RedisThreadSource


class MyEvent(BaseModel):
    name: str
    value: int


class TestRedisThreadSourceSubclass:
    def test_subclass_extracts_event_model(self) -> None:
        class MySrc(RedisThreadSource[MyEvent]):
            pass

        assert MySrc._event_model is MyEvent

    def test_subclass_without_type_param_raises(self) -> None:
        with pytest.raises(TypeError, match="must specify a concrete BaseModel type parameter"):

            class BadSrc(RedisThreadSource):  # type: ignore[type-arg]
                pass


class TestRedisProcessSourceSubclass:
    def test_subclass_extracts_event_model(self) -> None:
        class MySrc(RedisProcessSource[MyEvent]):
            pass

        assert MySrc._event_model is MyEvent

    def test_subclass_without_type_param_raises(self) -> None:
        with pytest.raises(TypeError, match="must specify a concrete BaseModel type parameter"):

            class BadSrc(RedisProcessSource):  # type: ignore[type-arg]
                pass
