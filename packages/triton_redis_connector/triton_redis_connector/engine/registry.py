"""Pipeline registry for type-annotation-based DAG construction.

Inspects __call__ signatures of registered nodes to build a directed
acyclic graph where union return types create multiple edges. Validates
graph integrity before the pipeline starts.
"""

from __future__ import annotations

import inspect
import sys
import types
import typing
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, TypeVar, overload

if TYPE_CHECKING:
    from collections.abc import Callable

from loguru import logger

from triton_redis_connector.engine.metric_aggregator import (
    InstrumentedNode,
    MetricAggregator,
    extract_event_counters,
)
from triton_redis_connector.protocols import (
    Processor,
    ProcessPoolProcessor,
    ProcessSource,
    Sink,
    Source,
    ThreadPoolProcessor,
    ThreadProcessor,
    ThreadSink,
    ThreadSource,
    WrappedNode,
)

_TIn = TypeVar("_TIn")
_TOut = TypeVar("_TOut")


class ExecutionVariant(StrEnum):
    """Execution mode for a pipeline node."""

    SOURCE = auto()
    THREAD_SOURCE = auto()
    PROCESS_SOURCE = auto()
    PROCESSOR = auto()
    THREAD_PROCESSOR = auto()
    THREAD_POOL_PROCESSOR = auto()
    PROCESS_POOL_PROCESSOR = auto()
    SINK = auto()
    THREAD_SINK = auto()


SOURCE_VARIANTS = {
    ExecutionVariant.SOURCE,
    ExecutionVariant.THREAD_SOURCE,
    ExecutionVariant.PROCESS_SOURCE,
}

SINK_VARIANTS = {
    ExecutionVariant.SINK,
    ExecutionVariant.THREAD_SINK,
}

INLINE_VARIANTS = {
    ExecutionVariant.SOURCE,
    ExecutionVariant.PROCESSOR,
    ExecutionVariant.SINK,
}


class GraphNode:
    """A node in the processing graph with its execution variant and type info."""

    def __init__(self, name: str, instance: Callable[..., Any], variant: ExecutionVariant) -> None:  # noqa: D107
        self.name = name
        self.instance = instance
        self.variant = variant
        self.input_types: set[type] = set()
        self.output_types: set[type] = set()


class PipelineRegistry:
    """Builds a DAG from registered pipeline nodes using type annotations.

    Inspects __call__ signatures to determine input/output event types.
    Union return types (A | B | None) create edges to consumers of both A and B.
    """

    def __init__(self) -> None:  # noqa: D107
        self._nodes: dict[str, GraphNode] = {}
        self._type_producers: dict[type, list[str]] = {}
        self._type_consumers: dict[type, list[str]] = {}
        self._aggregators: dict[str, MetricAggregator] = {}

    @overload
    def register(self, name: str, node: Source[_TOut]) -> None: ...
    @overload
    def register(self, name: str, node: Processor[_TIn, _TOut]) -> None: ...
    @overload
    def register(self, name: str, node: Sink[_TIn]) -> None: ...
    @overload
    def register(self, name: str, node: object) -> None: ...

    def register(self, name: str, node: Source[Any] | Processor[Any, Any] | Sink[Any] | object) -> None:
        """Register a pipeline node by inspecting its __call__ type hints."""
        variant = self._detect_variant(node)
        if not callable(node):
            msg = f"Node {name!r} is not callable"
            raise TypeError(msg)
        graph_node = GraphNode(name=name, instance=node, variant=variant)

        hints = self._get_call_hints(node)

        if variant not in SOURCE_VARIANTS:
            input_type = self._get_input_type(hints)
            if input_type is not None:
                graph_node.input_types = {input_type}
                self._type_consumers.setdefault(input_type, []).append(name)

        return_annotation = hints.get("return")
        output_types = self._extract_union_types(return_annotation, exclude={type(None)})
        graph_node.output_types = set(output_types)
        for out_type in output_types:
            self._type_producers.setdefault(out_type, []).append(name)

        self._nodes[name] = graph_node

        if isinstance(node, InstrumentedNode) and node.aggregator is not None:
            self._aggregators[name] = node.aggregator

        logger.debug(
            "Registered node {} as {} (inputs={}, outputs={})",
            name,
            variant,
            graph_node.input_types,
            graph_node.output_types,
        )

    def validate(self) -> list[str]:
        """Validate graph integrity. Returns a list of error messages (empty = valid).

        Type matching is subtype-aware: a consumer of ``BaseModel`` matches
        any producer of a ``BaseModel`` subclass, and a consumer of a
        ``Protocol`` matches producers of types that satisfy that protocol.
        """
        errors: list[str] = []

        for name, node in self._nodes.items():
            if not node.input_types and not node.output_types:
                errors.append(f"Orphan node '{name}' has no input or output types")

        all_produced_types = set(self._type_producers.keys())

        for consumed_type, consumers in self._type_consumers.items():
            if not self._any_producer_matches(consumed_type, all_produced_types):
                type_name = getattr(consumed_type, "__name__", str(consumed_type))
                errors.extend(
                    f"Node '{consumer_name}' consumes {type_name} but no node produces it"
                    for consumer_name in consumers
                )

        all_consumed_types = set(self._type_consumers.keys())

        for produced_type, producers in self._type_producers.items():
            if not self._any_consumer_matches(produced_type, all_consumed_types):
                for producer_name in producers:
                    node = self._nodes[producer_name]
                    if node.variant not in SINK_VARIANTS:
                        type_name = getattr(produced_type, "__name__", str(produced_type))
                        errors.append(f"Node '{producer_name}' produces {type_name} but no node consumes it")

        errors.extend(self._check_connectivity())

        return errors

    @staticmethod
    def _is_subtype(child: type, parent: type) -> bool:
        """Check if *child* is a subtype of *parent*, with protocol support.

        ``issubclass`` fails for protocols with non-method members (e.g.
        ``@property``).  When that happens, fall back to structural
        matching: check that all ``__protocol_attrs__`` exist on the
        child class or in its ``model_fields`` (pydantic v2).
        """
        try:
            return issubclass(child, parent)
        except TypeError:
            protocol_attrs: set[str] | None = getattr(parent, "__protocol_attrs__", None)
            if protocol_attrs is None:
                return False
            model_fields: set[str] = set(getattr(child, "model_fields", {}).keys())
            child_attrs = set(dir(child)) | model_fields
            return protocol_attrs <= child_attrs

    @classmethod
    def _any_producer_matches(cls, consumed_type: type, produced_types: set[type]) -> bool:
        """Check if any produced type is a subtype of consumed_type."""
        return any(produced == consumed_type or cls._is_subtype(produced, consumed_type) for produced in produced_types)

    @classmethod
    def _any_consumer_matches(cls, produced_type: type, consumed_types: set[type]) -> bool:
        """Check if any consumed type is a supertype of produced_type."""
        return any(produced_type == consumed or cls._is_subtype(produced_type, consumed) for consumed in consumed_types)

    def get_dispatch_order(self) -> list[GraphNode]:
        """Return inline nodes in topological order for main-loop dispatch.

        Only returns nodes with inline execution variants (Source, Processor, Sink).
        """
        inline_nodes = {name: node for name, node in self._nodes.items() if node.variant in INLINE_VARIANTS}
        if not inline_nodes:
            return []

        in_degree: dict[str, int] = dict.fromkeys(inline_nodes, 0)
        adjacency: dict[str, list[str]] = {name: [] for name in inline_nodes}

        for name, node in inline_nodes.items():
            for out_type in node.output_types:
                for consumer_name in self._type_consumers.get(out_type, []):
                    if consumer_name in inline_nodes:
                        adjacency[name].append(consumer_name)
                        in_degree[consumer_name] = in_degree.get(consumer_name, 0) + 1

        result: list[GraphNode] = []
        zero_in: list[str] = sorted([name for name, degree in in_degree.items() if degree == 0])

        while zero_in:
            current = zero_in.pop(0)
            result.append(inline_nodes[current])
            for neighbor in sorted(adjacency.get(current, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in.append(neighbor)
                    zero_in.sort()

        return result

    def get_node(self, name: str) -> GraphNode | None:
        """Return a registered node by name, or None if not found."""
        return self._nodes.get(name)

    @property
    def nodes(self) -> dict[str, GraphNode]:
        """Return all registered nodes."""
        return dict(self._nodes)

    @property
    def type_consumers(self) -> dict[type, list[str]]:
        """Return the type-to-consumer mapping for queue wiring."""
        return dict(self._type_consumers)

    @property
    def aggregators(self) -> dict[str, MetricAggregator]:
        """Return the node-name-to-aggregator mapping for metrics collection."""
        return dict(self._aggregators)

    @property
    def all_counter_names(self) -> set[str]:
        """Return all Counter metric names from event types in the graph.

        Scans ``_metrics_`` on all registered output event types.
        """
        names: set[str] = set()
        for event_type in self._type_producers:
            for counter in extract_event_counters(event_type):
                names.add(counter.name)
        return names

    def _check_connectivity(self) -> list[str]:
        """Check that the graph has no disconnected subgraphs.

        All nodes must be reachable from at least one source node.
        Uses subtype-aware matching: a produced type reaches consumers
        whose input type is a supertype or protocol satisfied by it.
        """
        errors: list[str] = []
        source_names = {name for name, node in self._nodes.items() if node.variant in SOURCE_VARIANTS}
        if not source_names and self._nodes:
            errors.append("No source nodes registered in the graph")
            return errors

        reachable: set[str] = set()
        frontier = list(source_names)

        while frontier:
            current_name = frontier.pop()
            if current_name in reachable:
                continue
            reachable.add(current_name)
            current_node = self._nodes[current_name]
            for out_type in current_node.output_types:
                for consumed_type, consumer_names in self._type_consumers.items():
                    if out_type == consumed_type or self._is_subtype(out_type, consumed_type):
                        frontier.extend(name for name in consumer_names if name not in reachable)

        unreachable = set(self._nodes.keys()) - reachable
        errors.extend(
            f"Node '{name}' is not reachable from any source (disconnected subgraph)" for name in sorted(unreachable)
        )

        return errors

    @staticmethod
    def _get_call_hints(node: object) -> dict[str, type]:
        """Get type hints from __call__, falling back to inner callable for wrappers.

        PEP 695 generic wrappers with ``from __future__ import annotations``
        have unresolved TypeVar forward references that cause NameError in
        ``typing.get_type_hints``. Resolution strategy:

        0. Explicit ``__call_hints__`` on the node or its ``_inner`` — set by the
           factory when it knows the concrete types (generic shipped processors)
        1. ``typing.get_type_hints`` on the outer node (works for concrete types)
        2. ``typing.get_type_hints`` on ``_inner`` (works when inner has no TypeVars)
        3. ``inspect.get_annotations(eval_str=True)`` on ``_inner`` with enriched
           globals — injects PEP 695 TypeVar bounds so ``T`` resolves to its bound
        """
        for target in (node, getattr(node, "_inner", None)):
            explicit = getattr(target, "__call_hints__", None)
            if isinstance(explicit, dict) and explicit:
                return explicit

        try:
            return typing.get_type_hints(node.__call__, include_extras=True)  # type: ignore[operator]
        except (NameError, AttributeError):
            inner = getattr(node, "_inner", None)
            if inner is not None:
                try:
                    return typing.get_type_hints(inner.__call__, include_extras=True)
                except (NameError, AttributeError):
                    pass
                return PipelineRegistry._eval_annotations_with_bounds(inner)
            return {}

    @staticmethod
    def _eval_annotations_with_bounds(node: object) -> dict[str, type]:
        """Resolve string annotations using module globals + PEP 695 TypeVar bounds.

        Builds a namespace from the node class's module and injects ``__bound__``
        for each TypeVar in ``__type_params__``, so ``T`` evaluates to its bound
        (e.g. ``BaseModel``).  Falls back to empty dict on any resolution failure.
        """
        cls = type(node)
        module = sys.modules.get(cls.__module__)
        if module is None:
            return {}
        ns: dict[str, Any] = vars(module).copy()
        for tp in getattr(cls, "__type_params__", ()):
            bound = getattr(tp, "__bound__", None)
            if bound is not None:
                ns[tp.__name__] = bound
        try:
            return inspect.get_annotations(node.__call__, globals=ns, eval_str=True)  # type: ignore[operator]
        except (NameError, AttributeError):
            return {}

    def _detect_variant(self, node: object) -> ExecutionVariant:
        """Detect the execution variant of a node.

        Fast path: if the node implements WrappedNode, use its declared variant.
        Fallback: uses call signature inspection to infer the variant.
        """
        if isinstance(node, WrappedNode):
            return ExecutionVariant(node.variant)

        if not callable(node):
            msg = f"Cannot detect execution variant for {type(node).__name__}"
            raise TypeError(msg)

        return self._infer_variant_from_signature(node)

    def _infer_variant_from_signature(self, node: object) -> ExecutionVariant:
        """Infer execution variant from __call__ parameter count and stop attribute."""
        hints = self._get_call_hints(node)
        param_count = len([k for k in hints if k != "return"])
        has_stop = self._has_stop_event(node)

        if param_count == 0:
            if has_stop:
                return ExecutionVariant.THREAD_SOURCE
            return ExecutionVariant.SOURCE

        return_type = self._unwrap_annotated(hints["return"]) if "return" in hints else None
        returns_none = return_type is type(None)

        if has_stop and returns_none:
            return ExecutionVariant.THREAD_SINK
        if has_stop:
            return ExecutionVariant.THREAD_PROCESSOR
        if returns_none:
            return ExecutionVariant.SINK
        return ExecutionVariant.PROCESSOR

    def _has_stop_event(self, node: object) -> bool:
        """Check if a node has a threading.Event stop attribute via protocol checks."""
        return isinstance(
            node,
            (ThreadSource, ProcessSource, ThreadProcessor, ThreadPoolProcessor, ProcessPoolProcessor, ThreadSink),
        )

    def _extract_union_types(
        self,
        annotation: type | None,
        exclude: set[type],
    ) -> list[type]:
        """Extract individual types from a union annotation, unwrapping Annotated.

        ``Annotated[A, ...] | B | None`` -> ``[A, B]``.
        Handles both ``typing.Union`` and ``|`` syntax.
        """
        if annotation is None:
            return []
        origin = typing.get_origin(annotation)
        if origin is types.UnionType or origin is typing.Union:
            result: list[type] = []
            for a in typing.get_args(annotation):
                base = self._unwrap_annotated(a)
                if base not in exclude:
                    result.append(base)
            return result
        base = self._unwrap_annotated(annotation)
        if base not in exclude:
            return [base]
        return []

    @staticmethod
    def _unwrap_annotated(annotation: type) -> type:
        """Unwrap ``Annotated[X, ...]`` to ``X``, or return annotation unchanged."""
        if typing.get_origin(annotation) is typing.Annotated:
            return typing.get_args(annotation)[0]  # type: ignore[no-any-return]
        return annotation

    def _get_input_type(self, hints: dict[str, type]) -> type | None:
        """Extract the input event type from __call__ hints.

        Looks for the first parameter after 'self' (named 'event' by convention,
        but matches any first non-return hint).
        """
        for param_name, param_type in hints.items():
            if param_name == "return":
                continue
            return param_type
        return None
