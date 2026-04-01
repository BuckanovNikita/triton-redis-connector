"""Tests for config-driven node assembly."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from triton_redis_connector.config.schema import NodeConfig, NodeVariant
from triton_redis_connector.engine.assembly import (
    assemble_all,
    assemble_node,
    import_class,
    resolve_type_params,
    wrap_with_variant,
)
from triton_redis_connector.engine.registry import ExecutionVariant, PipelineRegistry
from triton_redis_connector.engine.shipped_processors import EnvelopeExtractor
from triton_redis_connector.protocols import BaseThreadProcessor, Envelope, WrappedNode


class TestImportClass:
    def test_imports_shipped_processor(self) -> None:
        cls = import_class("triton_redis_connector.engine.shipped_processors.EnvelopeExtractor")
        assert cls.__name__ == "EnvelopeExtractor"

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
            import_class("EnvelopeExtractor")


class _StubEvent:
    pass


class _StubResult:
    pass


class _StubModel(BaseModel):
    value: str = "test"


class TestResolveTypeParams:
    def test_resolves_type_param_from_string(self) -> None:
        params = resolve_type_params(
            EnvelopeExtractor,
            {"model_class": "pydantic.BaseModel"},
        )
        assert params["model_class"] is BaseModel

    def test_passes_through_non_type_params(self) -> None:
        params = resolve_type_params(EnvelopeExtractor, {"model_class": BaseModel})
        assert params["model_class"] is BaseModel


class _PlainProcessor:
    def __call__(self, event: _StubEvent) -> _StubResult | None:
        return _StubResult()


class _BaseClassProcessor(BaseThreadProcessor[_StubEvent, _StubResult]):
    def __call__(self, event: _StubEvent) -> _StubResult | None:
        return _StubResult()


class TestWrapWithVariant:
    def test_wraps_plain_callable(self) -> None:
        inner = _PlainProcessor()
        wrapped = wrap_with_variant(inner, NodeVariant.THREAD_PROCESSOR, component_name="test")
        assert isinstance(wrapped, WrappedNode)
        assert wrapped.variant == ExecutionVariant.THREAD_PROCESSOR.value

    def test_skips_wrapping_matching_base_class(self) -> None:
        inner = _BaseClassProcessor()
        wrapped = wrap_with_variant(inner, NodeVariant.THREAD_PROCESSOR)
        assert wrapped is inner

    def test_returns_inline_unwrapped(self) -> None:
        inner = _PlainProcessor()
        wrapped = wrap_with_variant(inner, NodeVariant.PROCESSOR)
        assert wrapped is inner


class TestAssembleNode:
    def test_assembles_shipped_sink(self) -> None:
        cfg = NodeConfig(
            name="log_sink",
            class_path="triton_redis_connector.engine.shipped_processors.LogErrorSink",
            variant=NodeVariant.THREAD_SINK,
        )
        name, node = assemble_node(cfg)
        assert name == "log_sink"
        assert isinstance(node, WrappedNode)

    def test_assembles_envelope_extractor_with_type_param(self) -> None:
        cfg = NodeConfig(
            name="extractor",
            class_path="triton_redis_connector.engine.shipped_processors.EnvelopeExtractor",
            variant=NodeVariant.THREAD_PROCESSOR,
            params={"model_class": "pydantic.BaseModel"},
        )
        name, node = assemble_node(cfg)
        assert name == "extractor"
        assert isinstance(node, WrappedNode)


class TestAssembleAll:
    def test_registers_nodes_in_registry(self) -> None:
        registry = PipelineRegistry()

        class _Src:
            def __call__(self) -> Envelope[bytes] | None:
                return None

        registry.register("src", _Src())

        configs = [
            NodeConfig(
                name="extractor",
                class_path="triton_redis_connector.engine.shipped_processors.EnvelopeExtractor",
                variant=NodeVariant.THREAD_PROCESSOR,
                params={"model_class": "pydantic.BaseModel"},
            ),
        ]
        assemble_all(configs, registry)
        node = registry.get_node("extractor")
        assert node is not None
        assert node.variant == ExecutionVariant.THREAD_PROCESSOR

    def test_pool_size_creates_multiple_nodes(self) -> None:
        registry = PipelineRegistry()
        configs = [
            NodeConfig(
                name="worker",
                class_path="triton_redis_connector.engine.shipped_processors.LogErrorSink",
                variant=NodeVariant.THREAD_POOL_PROCESSOR,
                pool_size=3,
            ),
        ]
        assemble_all(configs, registry)
        assert registry.get_node("worker-0") is not None
        assert registry.get_node("worker-1") is not None
        assert registry.get_node("worker-2") is not None

    def test_empty_config_does_nothing(self) -> None:
        registry = PipelineRegistry()
        assemble_all([], registry)
        assert len(registry.nodes) == 0
