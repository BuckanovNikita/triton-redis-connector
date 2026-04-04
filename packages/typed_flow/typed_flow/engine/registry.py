"""Pipeline registry for type-annotation-based DAG construction.

Walks ``__orig_bases__`` of registered nodes to extract generic type
parameters for input/output types. Builds a directed acyclic graph
where union return types create multiple edges. Validates graph
integrity before the pipeline starts.
"""

from __future__ import annotations

import pickle
import types
import typing
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from loguru import logger

from typed_flow.engine.metric_aggregator import (
    InstrumentedNode,
    MetricAggregator,
    extract_event_counters,
)
from typed_flow.errors.models import PipelineError
from typed_flow.protocols import (
    BaseProcessPoolProcessor,
    BaseProcessSink,
    BaseProcessSource,
    BaseThreadPoolProcessor,
    BaseThreadProcessor,
    BaseThreadSink,
    BaseThreadSource,
    ProcessPoolProcessor,
    ProcessSink,
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
    PROCESS_SINK = auto()


SOURCE_VARIANTS = {
    ExecutionVariant.THREAD_SOURCE,
    ExecutionVariant.PROCESS_SOURCE,
}

SINK_VARIANTS = {
    ExecutionVariant.THREAD_SINK,
    ExecutionVariant.PROCESS_SINK,
}

POOL_VARIANTS = {
    ExecutionVariant.THREAD_POOL_PROCESSOR,
    ExecutionVariant.PROCESS_POOL_PROCESSOR,
}

_BASE_CLASS_TO_VARIANT: dict[type, ExecutionVariant] = {
    BaseThreadSource: ExecutionVariant.THREAD_SOURCE,
    BaseProcessSource: ExecutionVariant.PROCESS_SOURCE,
    BaseThreadProcessor: ExecutionVariant.THREAD_PROCESSOR,
    BaseThreadPoolProcessor: ExecutionVariant.THREAD_POOL_PROCESSOR,
    BaseProcessPoolProcessor: ExecutionVariant.PROCESS_POOL_PROCESSOR,
    BaseThreadSink: ExecutionVariant.THREAD_SINK,
    BaseProcessSink: ExecutionVariant.PROCESS_SINK,
}

_SOURCE_BASE_CLASSES = frozenset({BaseThreadSource, BaseProcessSource})
_PROCESSOR_BASE_CLASSES = frozenset({BaseThreadProcessor, BaseThreadPoolProcessor, BaseProcessPoolProcessor})
_SINK_BASE_CLASSES = frozenset({BaseThreadSink, BaseProcessSink})


class GraphNode:
    """A node in the processing graph with its execution variant and type info."""

    def __init__(
        self,
        name: str,
        instance: Callable[..., Any],
        variant: ExecutionVariant,
        num_workers: int = 1,
    ) -> None:
        """Initialize graph node with name, callable instance, and execution variant."""
        self.name = name
        self.instance = instance
        self.variant = variant
        self.num_workers = num_workers
        self.input_types: set[type] = set()
        self.output_types: set[type] = set()


def _unwrap_container(annotation: object) -> object:
    """Unwrap list[T] and tuple[T, ...] to T for type-based routing.

    The framework routes individual elements, so container wrappers
    are stripped at the type level. At runtime, _run_parallel_node
    unpacks list/tuple results and routes each element separately.
    """
    origin = typing.get_origin(annotation)
    if origin is list:
        args = typing.get_args(annotation)
        if args:
            return args[0]
    if origin is tuple:
        args = typing.get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
    return annotation


def _flatten_union(annotation: object) -> list[type]:
    """Flatten a union type (A | B | None) into individual types, excluding NoneType.

    Also unwraps list[T] and tuple[T, ...] containers to extract the
    element type T for routing.
    """
    annotation = _unwrap_container(annotation)
    origin = typing.get_origin(annotation)
    if origin is types.UnionType or origin is typing.Union:
        result: list[type] = []
        for arg in typing.get_args(annotation):
            if arg is type(None):
                continue
            unwrapped = _unwrap_annotated(_unwrap_container(arg))
            if unwrapped is not type(None):
                result.append(unwrapped)
        return result
    if isinstance(annotation, type) and annotation is not type(None):
        return [annotation]
    return []


def _is_subclass_safe(child: object, parent: object) -> bool:
    """Check issubclass, returning False for non-class types instead of raising."""
    if not isinstance(child, type) or not isinstance(parent, type):
        return False
    try:
        return issubclass(child, parent)
    except TypeError:
        return False


def _find_subclass_consumers(
    out_type: type,
    type_consumers: dict[type, list[str]],
) -> list[tuple[str, type]]:
    """Find consumers where out_type is a subclass of the consumed type.

    Skips exact matches (handled by dict lookup) and non-class types.
    Returns (consumer_name, consumed_type) pairs so the routing table
    stores the consumer's declared type for isinstance checks at runtime.
    """
    results: list[tuple[str, type]] = []
    for consumed_type, consumer_names in type_consumers.items():
        if consumed_type is out_type:
            continue
        if _is_subclass_safe(out_type, consumed_type):
            results.extend((name, consumed_type) for name in consumer_names)
    return results


def _unwrap_annotated(annotation: object) -> type:
    """Unwrap ``Annotated[X, ...]`` to ``X``, or return annotation unchanged."""
    if typing.get_origin(annotation) is typing.Annotated:
        return typing.get_args(annotation)[0]  # type: ignore[no-any-return]
    return annotation  # type: ignore[return-value]


def _build_typevar_map(cls: type) -> dict[typing.TypeVar, type]:
    """Build a TypeVar -> concrete type mapping from the class hierarchy.

    Walks ``__orig_bases__`` and matches generic parameters to their
    concrete arguments at each subclass level.
    """
    tv_map: dict[typing.TypeVar, type] = {}
    for klass in cls.__mro__:
        for base in getattr(klass, "__orig_bases__", ()):
            origin = typing.get_origin(base)
            if origin is None:
                continue
            args = typing.get_args(base)
            params: tuple[typing.TypeVar, ...] = getattr(origin, "__type_params__", ()) or getattr(
                origin, "__parameters__", ()
            )
            tv_map.update(
                {param: arg for param, arg in zip(params, args, strict=False) if not isinstance(arg, typing.TypeVar)}
            )
    return tv_map


def _resolve_pydantic_generic(annotation: object, tv_map: dict[typing.TypeVar, type]) -> object | None:
    """Substitute TypeVars in a Pydantic generic model via ``__pydantic_generic_metadata__``.

    Returns the re-parametrized class if substitution happened, otherwise None.
    """
    pydantic_meta: dict[str, Any] | None = getattr(annotation, "__pydantic_generic_metadata__", None)
    if pydantic_meta is None:
        return None
    pydantic_args: tuple[object, ...] = pydantic_meta.get("args", ())
    pydantic_origin: type | None = pydantic_meta.get("origin")
    if not pydantic_args or pydantic_origin is None:
        return None
    new_args = tuple(_substitute_typevars(a, tv_map) for a in pydantic_args)
    if new_args == pydantic_args:
        return None
    return pydantic_origin[new_args]  # type: ignore[index,no-any-return]


def _substitute_typevars(annotation: object, tv_map: dict[typing.TypeVar, type]) -> object:
    """Recursively substitute TypeVars in a type annotation using the given map.

    Handles standard typing generics (via ``get_origin``/``get_args``) and
    Pydantic generic models (via ``_resolve_pydantic_generic``).
    """
    if isinstance(annotation, typing.TypeVar):
        return tv_map.get(annotation, annotation)

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is None and not args:
        return _resolve_pydantic_generic(annotation, tv_map) or annotation

    if not args:
        return annotation

    new_args = tuple(_substitute_typevars(a, tv_map) for a in args)
    if new_args == args:
        return annotation

    if origin is types.UnionType:
        result: object = new_args[0]
        for a in new_args[1:]:
            result = result | a  # type: ignore[operator]
        return result

    return origin[new_args]  # type: ignore[index]


def _extract_types_from_generics(node: object) -> tuple[set[type], set[type]]:
    """Extract (input_types, output_types) from a node's ``__orig_bases__``.

    Walks the class hierarchy to find the Base* class (BaseThreadSource,
    BaseThreadProcessor, etc.) and extracts its generic type parameters.
    Resolves TypeVars using the concrete type arguments from subclasses.

    Sources: args[0] = output type (may be union)
    Processors: args[0] = input, args[1] = output (each may be union)
    Sinks: args[0] = input type (may be union)
    """
    cls = type(node)
    tv_map = _build_typevar_map(cls)

    for klass in cls.__mro__:
        for base in getattr(klass, "__orig_bases__", ()):
            origin = typing.get_origin(base)
            if origin is None:
                continue
            args = typing.get_args(base)
            if not args:
                continue

            if origin in _SOURCE_BASE_CLASSES:
                resolved = _substitute_typevars(args[0], tv_map)
                return set(), set(_flatten_union(resolved))

            min_args_for_processor = 2
            if origin in _PROCESSOR_BASE_CLASSES and len(args) >= min_args_for_processor:
                return set(_flatten_union(args[0])), set(_flatten_union(args[1]))

            if origin in _SINK_BASE_CLASSES:
                resolved = _substitute_typevars(args[0], tv_map)
                return set(_flatten_union(resolved)), set()

    return set(), set()


class PipelineRegistry:
    """Builds a DAG from registered pipeline nodes using type annotations.

    Walks ``__orig_bases__`` to determine input/output event types.
    Union return types (A | B | None) create edges to consumers of both A and B.
    """

    def __init__(self) -> None:
        """Initialize empty registry with type tracking maps."""
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

        num_workers = 1
        if variant in POOL_VARIANTS:
            num_workers = node.num_workers  # type: ignore[attr-defined]
            if variant == ExecutionVariant.PROCESS_POOL_PROCESSOR:
                _validate_picklable(name, node)

        graph_node = GraphNode(name=name, instance=node, variant=variant, num_workers=num_workers)

        input_types, output_types = _extract_types_from_generics(node)

        if variant not in SOURCE_VARIANTS:
            graph_node.input_types = input_types
            for in_type in input_types:
                self._type_consumers.setdefault(in_type, []).append(name)

        consumes_errors = any(issubclass(PipelineError, t) for t in input_types)
        if not consumes_errors and PipelineError not in output_types:
            output_types = output_types | {PipelineError}
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

        Uses subtype-aware matching: a produced type satisfies a consumer
        if the produced type is a subclass of the consumed type.
        """
        errors: list[str] = []

        for name, node in self._nodes.items():
            if not node.input_types and not node.output_types:
                errors.append(f"Orphan node '{name}' has no input or output types")

        all_produced_types = set(self._type_producers.keys())

        for consumed_type, consumers in self._type_consumers.items():
            has_producer = any(_is_subclass_safe(pt, consumed_type) for pt in all_produced_types)
            if not has_producer:
                type_name = _type_display_name(consumed_type)
                errors.extend(
                    f"Node '{consumer_name}' consumes {type_name} but no node produces it"
                    for consumer_name in consumers
                )

        all_consumed_types = set(self._type_consumers.keys())

        for produced_type, producers in self._type_producers.items():
            has_consumer = any(_is_subclass_safe(produced_type, ct) for ct in all_consumed_types)
            if not has_consumer:
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
        Uses subtype-aware matching for connectivity.
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
                consumer_names = list(self._type_consumers.get(out_type, []))
                for name, _ in _find_subclass_consumers(out_type, self._type_consumers):
                    if name not in consumer_names:
                        consumer_names.append(name)
                frontier.extend(name for name in consumer_names if name not in reachable)

        unreachable = set(self._nodes.keys()) - reachable
        errors.extend(
            f"Node '{name}' is not reachable from any source (disconnected subgraph)" for name in sorted(unreachable)
        )

        return errors

    def compute_routing_table(self) -> dict[str, list[tuple[str, type]]]:
        """Build a producer-name -> list[(consumer_name, expected_input_type)] table.

        Uses subtype-aware matching. For exact matches the producer's output
        type is stored; for subclass matches the consumer's declared input
        type is stored so isinstance checks work at runtime.
        """
        table: dict[str, list[tuple[str, type]]] = {}
        for name, node in self._nodes.items():
            consumer_pairs: list[tuple[str, type]] = []
            for out_type in node.output_types:
                consumer_pairs.extend(
                    (consumer_name, out_type) for consumer_name in self._type_consumers.get(out_type, [])
                )
                consumer_pairs.extend(_find_subclass_consumers(out_type, self._type_consumers))
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
        (ProcessSink, ExecutionVariant.PROCESS_SINK),
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
            f"ProcessPoolProcessor, ThreadSink, ProcessSink)"
        )
        raise TypeError(msg)


def _validate_picklable(name: str, node: object) -> None:
    """Verify that a process-pool node can survive pickle round-trip.

    Called at registration time so users get an immediate, clear error
    instead of a cryptic PicklingError at runtime.
    """
    try:
        pickle.loads(pickle.dumps(node))  # noqa: S301
    except (pickle.PicklingError, TypeError, AttributeError) as exc:
        msg = (
            f"Process-pool node {name!r} ({type(node).__name__}) is not picklable. "
            f"Store only picklable config values in __init__ and lazy-init "
            f"heavy objects on first __call__: {exc}"
        )
        raise TypeError(msg) from exc


def _type_display_name(t: type) -> str:
    """Human-readable name for a type, including generic args."""
    return getattr(t, "__name__", str(t))
