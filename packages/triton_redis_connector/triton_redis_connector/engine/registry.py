"""Pipeline registry for type-annotation-based DAG construction.

Inspects __call__ signatures of registered nodes to build a directed
acyclic graph where union return types create multiple edges. Validates
graph integrity before the pipeline starts.
"""

from __future__ import annotations

import types
import typing
from enum import StrEnum, auto
from typing import Any, TypeVar, overload

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

    def __init__(self, name: str, instance: object, variant: ExecutionVariant) -> None:  # noqa: D107
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
        graph_node = GraphNode(name=name, instance=node, variant=variant)

        call_method = node.__call__  # type: ignore[operator]
        hints = typing.get_type_hints(call_method, include_extras=True)

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
        """Validate graph integrity. Returns a list of error messages (empty = valid)."""
        errors: list[str] = []

        for name, node in self._nodes.items():
            if not node.input_types and not node.output_types:
                errors.append(f"Orphan node '{name}' has no input or output types")

        for event_type, consumers in self._type_consumers.items():
            if event_type not in self._type_producers:
                errors.extend(
                    f"Node '{consumer_name}' consumes {event_type.__name__} but no node produces it"
                    for consumer_name in consumers
                )

        for event_type, producers in self._type_producers.items():
            if event_type not in self._type_consumers:
                for producer_name in producers:
                    node = self._nodes[producer_name]
                    if node.variant not in SINK_VARIANTS:
                        errors.append(f"Node '{producer_name}' produces {event_type.__name__} but no node consumes it")

        errors.extend(self._check_connectivity())

        return errors

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
                frontier.extend(
                    consumer_name
                    for consumer_name in self._type_consumers.get(out_type, [])
                    if consumer_name not in reachable
                )

        unreachable = set(self._nodes.keys()) - reachable
        errors.extend(
            f"Node '{name}' is not reachable from any source (disconnected subgraph)" for name in sorted(unreachable)
        )

        return errors

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
        call_method = node.__call__  # type: ignore[operator]
        hints = typing.get_type_hints(call_method, include_extras=True)
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
