"""Pipeline registry for type-annotation-based DAG construction.

Walks ``__orig_bases__`` of registered nodes to extract generic type
parameters for input/output types. Builds a directed acyclic graph
where union return types create multiple edges. Validates graph
integrity before the pipeline starts.
"""

from __future__ import annotations

import types
import typing
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from loguru import logger

from triton_redis_connector.engine.metric_aggregator import (
    InstrumentedNode,
    MetricAggregator,
    extract_event_counters,
)
from triton_redis_connector.protocols import (
    BaseProcessPoolProcessor,
    BaseProcessSource,
    BaseThreadPoolProcessor,
    BaseThreadProcessor,
    BaseThreadSink,
    BaseThreadSource,
    ProcessPoolProcessor,
    ProcessSource,
    ThreadPoolProcessor,
    ThreadProcessor,
    ThreadSink,
    ThreadSource,
    WrappedNode,
)


class ExecutionVariant(StrEnum):
    """Execution mode for a pipeline node (parallel variants only)."""

    THREAD_SOURCE = auto()
    PROCESS_SOURCE = auto()
    THREAD_PROCESSOR = auto()
    THREAD_POOL_PROCESSOR = auto()
    PROCESS_POOL_PROCESSOR = auto()
    THREAD_SINK = auto()


SOURCE_VARIANTS = {
    ExecutionVariant.THREAD_SOURCE,
    ExecutionVariant.PROCESS_SOURCE,
}

SINK_VARIANTS = {
    ExecutionVariant.THREAD_SINK,
}

_BASE_CLASS_TO_VARIANT: dict[type, ExecutionVariant] = {
    BaseThreadSource: ExecutionVariant.THREAD_SOURCE,
    BaseProcessSource: ExecutionVariant.PROCESS_SOURCE,
    BaseThreadProcessor: ExecutionVariant.THREAD_PROCESSOR,
    BaseThreadPoolProcessor: ExecutionVariant.THREAD_POOL_PROCESSOR,
    BaseProcessPoolProcessor: ExecutionVariant.PROCESS_POOL_PROCESSOR,
    BaseThreadSink: ExecutionVariant.THREAD_SINK,
}

_SOURCE_BASE_CLASSES = frozenset({BaseThreadSource, BaseProcessSource})
_PROCESSOR_BASE_CLASSES = frozenset({BaseThreadProcessor, BaseThreadPoolProcessor, BaseProcessPoolProcessor})
_SINK_BASE_CLASSES = frozenset({BaseThreadSink})


class GraphNode:
    """A node in the processing graph with its execution variant and type info."""

    def __init__(self, name: str, instance: Callable[..., Any], variant: ExecutionVariant) -> None:  # noqa: D107
        self.name = name
        self.instance = instance
        self.variant = variant
        self.input_types: set[type] = set()
        self.output_types: set[type] = set()


def _flatten_union(annotation: object) -> list[type]:
    """Flatten a union type (A | B | None) into individual types, excluding NoneType."""
    origin = typing.get_origin(annotation)
    if origin is types.UnionType or origin is typing.Union:
        result: list[type] = []
        for arg in typing.get_args(annotation):
            if arg is type(None):
                continue
            unwrapped = _unwrap_annotated(arg)
            if unwrapped is not type(None):
                result.append(unwrapped)
        return result
    if isinstance(annotation, type) and annotation is not type(None):
        return [annotation]
    return []


def _unwrap_annotated(annotation: object) -> type:
    """Unwrap ``Annotated[X, ...]`` to ``X``, or return annotation unchanged."""
    if typing.get_origin(annotation) is typing.Annotated:
        return typing.get_args(annotation)[0]  # type: ignore[no-any-return]
    return annotation  # type: ignore[return-value]


def _extract_types_from_generics(node: object) -> tuple[set[type], set[type]]:
    """Extract (input_types, output_types) from a node's ``__orig_bases__``.

    Walks the class hierarchy to find the Base* class (BaseThreadSource,
    BaseThreadProcessor, etc.) and extracts its generic type parameters.

    Sources: args[0] = output type (may be union)
    Processors: args[0] = input, args[1] = output (each may be union)
    Sinks: args[0] = input type (may be union)
    """
    cls = type(node)
    for klass in cls.__mro__:
        for base in getattr(klass, "__orig_bases__", ()):
            origin = typing.get_origin(base)
            if origin is None:
                continue
            args = typing.get_args(base)
            if not args:
                continue

            if origin in _SOURCE_BASE_CLASSES:
                return set(), set(_flatten_union(args[0]))

            min_args_for_processor = 2
            if origin in _PROCESSOR_BASE_CLASSES and len(args) >= min_args_for_processor:
                return set(_flatten_union(args[0])), set(_flatten_union(args[1]))

            if origin in _SINK_BASE_CLASSES:
                return set(_flatten_union(args[0])), set()

    return set(), set()


class PipelineRegistry:
    """Builds a DAG from registered pipeline nodes using type annotations.

    Walks ``__orig_bases__`` to determine input/output event types.
    Union return types (A | B | None) create edges to consumers of both A and B.
    """

    def __init__(self) -> None:  # noqa: D107
        self._nodes: dict[str, GraphNode] = {}
        self._type_producers: dict[type, list[str]] = {}
        self._type_consumers: dict[type, list[str]] = {}
        self._aggregators: dict[str, MetricAggregator] = {}

    def register(self, name: str, node: object) -> None:
        """Register a pipeline node by inspecting its ``__orig_bases__`` type parameters."""
        variant = self._detect_variant(node)
        if not callable(node):
            msg = f"Node {name!r} is not callable"
            raise TypeError(msg)
        graph_node = GraphNode(name=name, instance=node, variant=variant)

        input_types, output_types = _extract_types_from_generics(node)

        if variant not in SOURCE_VARIANTS:
            graph_node.input_types = input_types
            for in_type in input_types:
                self._type_consumers.setdefault(in_type, []).append(name)

        graph_node.output_types = output_types
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

        Uses exact type matching: types must match exactly (no subtype checking).
        """
        errors: list[str] = []

        for name, node in self._nodes.items():
            if not node.input_types and not node.output_types:
                errors.append(f"Orphan node '{name}' has no input or output types")

        all_produced_types = set(self._type_producers.keys())

        for consumed_type, consumers in self._type_consumers.items():
            if consumed_type not in all_produced_types:
                type_name = _type_display_name(consumed_type)
                errors.extend(
                    f"Node '{consumer_name}' consumes {type_name} but no node produces it"
                    for consumer_name in consumers
                )

        all_consumed_types = set(self._type_consumers.keys())

        for produced_type, producers in self._type_producers.items():
            if produced_type not in all_consumed_types:
                for producer_name in producers:
                    node = self._nodes[producer_name]
                    if node.variant not in SINK_VARIANTS:
                        type_name = _type_display_name(produced_type)
                        errors.append(f"Node '{producer_name}' produces {type_name} but no node consumes it")

        errors.extend(self._check_connectivity())

        return errors

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
        Uses exact type matching for connectivity.
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
                consumer_names = self._type_consumers.get(out_type, [])
                frontier.extend(name for name in consumer_names if name not in reachable)

        unreachable = set(self._nodes.keys()) - reachable
        errors.extend(
            f"Node '{name}' is not reachable from any source (disconnected subgraph)" for name in sorted(unreachable)
        )

        return errors

    def compute_routing_table(self) -> dict[str, list[tuple[str, type]]]:
        """Build a producer-name -> list[(consumer_name, expected_input_type)] table.

        Uses exact type matching. Each entry tells the router which consumer
        to consider and what type to check against at runtime.
        """
        table: dict[str, list[tuple[str, type]]] = {}
        for name, node in self._nodes.items():
            consumer_pairs: list[tuple[str, type]] = []
            for out_type in node.output_types:
                consumer_pairs.extend(
                    (consumer_name, out_type) for consumer_name in self._type_consumers.get(out_type, [])
                )
            if consumer_pairs:
                seen: set[tuple[str, type]] = set()
                unique: list[tuple[str, type]] = []
                for pair in consumer_pairs:
                    if pair not in seen:
                        seen.add(pair)
                        unique.append(pair)
                table[name] = unique
        return table

    _PROTOCOL_TO_VARIANT: typing.ClassVar[list[tuple[type, ExecutionVariant]]] = [
        (ProcessSource, ExecutionVariant.PROCESS_SOURCE),
        (ThreadSource, ExecutionVariant.THREAD_SOURCE),
        (ProcessPoolProcessor, ExecutionVariant.PROCESS_POOL_PROCESSOR),
        (ThreadPoolProcessor, ExecutionVariant.THREAD_POOL_PROCESSOR),
        (ThreadProcessor, ExecutionVariant.THREAD_PROCESSOR),
        (ThreadSink, ExecutionVariant.THREAD_SINK),
    ]

    def _detect_variant(self, node: object) -> ExecutionVariant:
        """Detect the execution variant of a node.

        If the node implements WrappedNode, use its declared variant.
        Otherwise checks protocol conformance for parallel variants.
        Raises TypeError if no variant can be determined.
        """
        if isinstance(node, WrappedNode):
            return ExecutionVariant(node.variant)

        if not callable(node):
            msg = f"Cannot detect execution variant for {type(node).__name__}: not callable"
            raise TypeError(msg)

        for protocol_cls, variant in self._PROTOCOL_TO_VARIANT:
            if isinstance(node, protocol_cls):
                return variant

        msg = (
            f"Cannot detect execution variant for {type(node).__name__}: "
            f"node must implement WrappedNode or a parallel protocol "
            f"(ThreadSource, ProcessSource, ThreadProcessor, ThreadPoolProcessor, "
            f"ProcessPoolProcessor, ThreadSink)"
        )
        raise TypeError(msg)


def _type_display_name(t: type) -> str:
    """Human-readable name for a type, including generic args."""
    return getattr(t, "__name__", str(t))
