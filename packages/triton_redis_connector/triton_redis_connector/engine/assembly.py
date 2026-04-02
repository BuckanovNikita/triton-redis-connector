"""Config-driven node assembly: import, instantiate, wrap, register.

Resolves pipeline nodes from ``NodeConfig`` entries in the application config.
Shipped processors remain variant-agnostic callables — the variant is selected
at assembly time via the config and applied through framework wrappers.
User processors that inherit from base classes already declare their variant
and are registered directly (no wrapping).

User processors use ``class_path`` in NodeConfig. Shipped processors are
registered automatically by ``build_application`` based on AppConfig fields.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any

from loguru import logger

from triton_redis_connector.config.schema import NodeVariant
from triton_redis_connector.engine.metric_aggregator import MetricAggregator
from triton_redis_connector.engine.registry import ExecutionVariant
from triton_redis_connector.engine.wrappers import (
    DetachedProcessSource,
    DetachedThreadProcessor,
    DetachedThreadSink,
    DetachedThreadSource,
    MultiProcessProcessor,
    ThreadedProcessor,
)
from triton_redis_connector.protocols import WrappedNode

if TYPE_CHECKING:
    from triton_redis_connector.config.schema import NodeConfig
    from triton_redis_connector.engine.registry import PipelineRegistry
    from triton_redis_connector.protocols import MetricRecorder

_VARIANT_MAP: dict[NodeVariant, ExecutionVariant] = {
    NodeVariant.THREAD_SOURCE: ExecutionVariant.THREAD_SOURCE,
    NodeVariant.PROCESS_SOURCE: ExecutionVariant.PROCESS_SOURCE,
    NodeVariant.THREAD_PROCESSOR: ExecutionVariant.THREAD_PROCESSOR,
    NodeVariant.THREAD_POOL_PROCESSOR: ExecutionVariant.THREAD_POOL_PROCESSOR,
    NodeVariant.PROCESS_POOL_PROCESSOR: ExecutionVariant.PROCESS_POOL_PROCESSOR,
    NodeVariant.THREAD_SINK: ExecutionVariant.THREAD_SINK,
}


def import_class(class_path: str) -> type:
    """Import a class from a fully-qualified dotted path.

    Raises ``ImportError`` if the module cannot be found and ``AttributeError``
    if the class does not exist in the module.
    """
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        msg = f"class_path must be fully-qualified (got {class_path!r})"
        raise ImportError(msg)
    module = importlib.import_module(module_path)
    result: type = getattr(module, class_name)
    return result


def resolve_type_params(cls: type, raw_params: dict[str, Any]) -> dict[str, Any]:
    """Resolve string class paths in params that map to ``type[X]`` hints.

    Inspects the ``__init__`` signature of *cls*. For each parameter annotated
    as ``type[X]``, if the raw value is a string, import it as a class.
    Other parameters pass through unchanged.
    """
    type_param_names = _get_init_type_params(cls)
    resolved: dict[str, Any] = {}
    for key, value in raw_params.items():
        if isinstance(value, str) and key in type_param_names:
            resolved[key] = import_class(value)
        else:
            resolved[key] = value
    return resolved


def wrap_with_variant(
    inner: object,
    target_variant: NodeVariant,
    *,
    aggregator: MetricAggregator | None = None,
    metric_recorder: MetricRecorder | None = None,
    component_name: str = "",
) -> object:
    """Wrap a callable node with the appropriate execution variant wrapper.

    If *inner* already satisfies ``WrappedNode`` and declares the correct
    variant, it is returned as-is (idempotent). Otherwise wraps with the
    framework wrapper for the target parallel variant.
    """
    if isinstance(inner, WrappedNode):
        declared = ExecutionVariant(inner.variant)
        expected = _VARIANT_MAP[target_variant]
        if declared == expected:
            return inner

    wrapper_cls = _get_wrapper_class(target_variant)
    return wrapper_cls(inner, aggregator=aggregator, metric_recorder=metric_recorder, component_name=component_name)


def assemble_node(
    node_config: NodeConfig,
    *,
    metric_recorder: MetricRecorder | None = None,
) -> tuple[str, object]:
    """Assemble a single pipeline node from its config entry.

    Returns ``(name, wrapped_instance)`` ready for ``registry.register()``.
    """
    cls = import_class(node_config.class_path)
    params = resolve_type_params(cls, dict(node_config.params))
    instance = cls(**params)

    aggregator = MetricAggregator() if node_config.tracked else None
    component_name = node_config.tracked_name or node_config.name

    wrapped = wrap_with_variant(
        instance,
        node_config.variant,
        aggregator=aggregator,
        metric_recorder=metric_recorder,
        component_name=component_name,
    )

    logger.debug(
        "Assembled node {} (class={}, variant={})",
        node_config.name,
        node_config.class_path,
        node_config.variant,
    )
    return node_config.name, wrapped


def assemble_all(
    nodes: list[NodeConfig],
    registry: PipelineRegistry,
    *,
    metric_recorder: MetricRecorder | None = None,
) -> None:
    """Assemble all config-driven nodes and register them in the pipeline."""
    for node_config in nodes:
        pool_size = node_config.pool_size
        if pool_size > 1:
            for i in range(pool_size):
                name, wrapped = assemble_node(node_config, metric_recorder=metric_recorder)
                registry.register(f"{name}-{i}", wrapped)
        else:
            name, wrapped = assemble_node(node_config, metric_recorder=metric_recorder)
            registry.register(name, wrapped)
    if nodes:
        logger.info("Assembled {} config-driven node(s)", len(nodes))


def _get_wrapper_class(target_variant: NodeVariant) -> type:
    """Map a non-inline NodeVariant to its wrapper class."""
    mapping: dict[NodeVariant, type] = {
        NodeVariant.THREAD_SOURCE: DetachedThreadSource,
        NodeVariant.PROCESS_SOURCE: DetachedProcessSource,
        NodeVariant.THREAD_PROCESSOR: DetachedThreadProcessor,
        NodeVariant.THREAD_POOL_PROCESSOR: ThreadedProcessor,
        NodeVariant.PROCESS_POOL_PROCESSOR: MultiProcessProcessor,
        NodeVariant.THREAD_SINK: DetachedThreadSink,
    }
    cls = mapping.get(target_variant)
    if cls is None:
        msg = f"No wrapper for variant {target_variant!r}"
        raise ValueError(msg)
    return cls


def _get_init_type_params(cls: type) -> set[str]:
    """Return parameter names whose annotation looks like ``type[...]``.

    Works with both evaluated and string annotations (PEP 695 + future
    annotations). For string annotations, matches the pattern ``type[X]``.
    For evaluated annotations, checks ``typing.get_origin(...) is type``.
    """
    raw = inspect.get_annotations(cls.__init__)  # type: ignore[misc]
    result: set[str] = set()
    for name, annotation in raw.items():
        if name == "return":
            continue
        if isinstance(annotation, str) and annotation.startswith("type["):
            result.add(name)
    return result
