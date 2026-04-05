"""Tests for Redis sink serialization logic."""

from pydantic import BaseModel
from typed_flow.connectors.redis_sink import _serialize_to_dict
from typed_flow.protocols.messaging import FrameworkMeta


class PlainModel(BaseModel):
    name: str
    count: int


class ModelWithFlowMeta(BaseModel):
    name: str
    count: int
    flow_meta: FrameworkMeta


class TestSerializeToDict:
    def test_serialize_plain_model_all_fields_as_bytes(self) -> None:
        model = PlainModel(name="test", count=42)
        result = _serialize_to_dict(model)
        assert result["name"] == b"test"
        assert result["count"] == b"42"

    def test_serialize_strips_flow_meta(self) -> None:
        meta = FrameworkMeta(source_stream="s", message_id="1-0", priority=0)
        model = ModelWithFlowMeta(name="test", count=1, flow_meta=meta)
        result = _serialize_to_dict(model)
        assert "flow_meta" not in result
        assert "name" in result

    def test_serialize_without_flow_meta_keeps_all_fields(self) -> None:
        model = PlainModel(name="hello", count=99)
        result = _serialize_to_dict(model)
        assert len(result) == 2
        assert "name" in result
        assert "count" in result
