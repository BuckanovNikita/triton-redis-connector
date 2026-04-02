"""Tests for config-driven node assembly utilities."""

from __future__ import annotations

import pytest
from triton_redis_connector.engine.assembly import (
    import_class,
    resolve_type_params,
)


class TestImportClass:
    def test_imports_shipped_processor(self) -> None:
        cls = import_class("triton_redis_connector.engine.shipped_processors.LogErrorSinkNode")
        assert cls.__name__ == "LogErrorSinkNode"

    def test_imports_builtin_type(self) -> None:
        cls = import_class("pydantic.BaseModel")
        assert cls.__name__ == "BaseModel"

    def test_raises_on_missing_module(self) -> None:
        with pytest.raises(ImportError):
            import_class("nonexistent.module.Class")

    def test_raises_on_missing_class(self) -> None:
        with pytest.raises(AttributeError):
            import_class("triton_redis_connector.engine.shipped_processors.NonExistent")

    def test_raises_on_bare_name(self) -> None:
        with pytest.raises(ImportError, match="fully-qualified"):
            import_class("LogErrorSinkNode")


class _TypeParamClass:
    def __init__(self, event_type: type[object] | None = None) -> None:
        self.event_type = event_type


class TestResolveTypeParams:
    def test_resolves_type_param_from_string(self) -> None:
        params = resolve_type_params(
            _TypeParamClass,
            {"event_type": "pydantic.BaseModel"},
        )
        from pydantic import BaseModel

        assert params["event_type"] is BaseModel

    def test_passes_through_non_type_params(self) -> None:
        from pydantic import BaseModel

        params = resolve_type_params(_TypeParamClass, {"event_type": BaseModel})
        assert params["event_type"] is BaseModel
