"""Application skeleton -- main entry point for the pipeline.

The Application class owns the main loop, monitoring parallel variant
health and managing graceful shutdown. All nodes run in dedicated threads.
Resolved from the dishka container via EngineProvider, or via the
``Application.build()`` classmethod for explicit processor-list wiring.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import redis as redis_lib
from loguru import logger

from typed_flow.engine.diagram import generate_mermaid
from typed_flow.engine.queue_bus import QueueBus
from typed_flow.engine.registry import SINK_VARIANTS, SOURCE_VARIANTS, PipelineRegistry
from typed_flow.engine.validation import validate_graph
from typed_flow.errors.models import ConnectorError
from typed_flow.metrics.buffer import MetricBuffer
from typed_flow.metrics.emitter import MetricEmitter
from typed_flow.protocols import (
    LifecycleService,
    MetricRecorder,
    ProcessPoolProcessor,
    ProcessSource,
    ThreadPoolProcessor,
    ThreadProcessor,
    ThreadSink,
    ThreadSource,
)

if TYPE_CHECKING:
    from typed_flow.config.schema import AppConfig
    from typed_flow.engine.registry import GraphNode


def _pydantic_generic_match(result_type: type, expected_type: type) -> bool:
    """Compare Pydantic parametrized generic classes by their origin and args.

    Pydantic's ``__class_getitem__`` cache is not thread-safe — calling
    ``Model[T]`` from different threads may produce distinct class objects.
    When ``is`` identity fails, this function compares via
    ``__pydantic_generic_metadata__`` (origin + args) as a fallback.
    """
    result_meta: dict[str, object] | None = getattr(result_type, "__pydantic_generic_metadata__", None)
    expected_meta: dict[str, object] | None = getattr(expected_type, "__pydantic_generic_metadata__", None)
    if result_meta is None or expected_meta is None:
        return False
    return result_meta.get("origin") is expected_meta.get("origin") and result_meta.get("args") == expected_meta.get(
        "args"
    )


class Application:
    """Main pipeline application.

    All nodes run as parallel variants in dedicated threads. The main
    loop monitors variant health and handles graceful shutdown.
    """

    def __init__(  # noqa: PLR0913
        self,
        config: AppConfig,
        registry: PipelineRegistry,
        queue_bus: QueueBus,
        extra_nodes: list[tuple[str, object]] | None = None,
        metric_buffer: MetricRecorder | None = None,
        extra_service: LifecycleService | None = None,
        metric_emitter: LifecycleService | None = None,
    ) -> None:
        """Initialize Application with config, registry, queue bus, and optional services."""
        self._config = config
        self._registry = registry
        self._queue_bus = queue_bus
        self._shutdown = threading.Event()
        self._parallel_threads: dict[str, threading.Thread] = {}
        if extra_nodes:
            for name, node in extra_nodes:
                self._registry.register(name, node)
        self._restart_counts: dict[str, int] = {}
        self._metric_buffer = metric_buffer
        self._extra_service = extra_service
        self._metric_emitter = metric_emitter
        self._stats_processed: int = 0
        self._stats_in_progress: int = 0
        self._stats_errors: int = 0
        self._stats_per_node: dict[str, int] = {}
        self._stats_sink_writes: dict[str, int] = {}
        self._stats_timer: threading.Timer | None = None
        self._stats_interval: float = 0.0
        self._routing_table: dict[str, list[tuple[str, type]]] = {}

    @classmethod
    def build(
        cls,
        config: AppConfig,
        processors: list[object],
        *,
        redis_client: redis_lib.Redis[bytes] | None = None,
        extra_service: LifecycleService | None = None,
    ) -> Application:
        """Build a fully wired Application from an explicit processor list.

        Each processor instance is registered using its ``name`` attribute
        (if present) or its class name. Auto-creates MetricBuffer,
        MetricEmitter, QueueBus, and PipelineRegistry from config.
        Runs graph validation before returning.
        """
        registry = PipelineRegistry()
        used_names: dict[str, int] = {}

        for proc in processors:
            raw_name: str = _node_name(proc)
            count = used_names.get(raw_name, 0)
            used_names[raw_name] = count + 1
            name = raw_name if count == 0 else f"{raw_name}_{count}"
            registry.register(name, proc)

        errors = validate_graph(registry)
        if errors:
            for error in errors:
                logger.error("Build validation error: {}", error)
            msg = f"Pipeline graph has {len(errors)} validation error(s)"
            raise ValueError(msg)

        queue_bus = QueueBus(config=config.queues)
        metric_buffer = MetricBuffer()

        metric_emitter: LifecycleService | None = None
        if config.metrics.output_stream:
            metrics_client = redis_client or redis_lib.Redis(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                decode_responses=False,
            )
            metric_emitter = MetricEmitter(
                buffer=metric_buffer,
                client=metrics_client,
                stream=config.metrics.output_stream,
                interval_s=config.metrics.emit_interval_s,
                known_metrics=registry.all_counter_names,
                app_name=os.environ.get("APP_NAME", ""),
            )

        return cls(
            config=config,
            registry=registry,
            queue_bus=queue_bus,
            metric_buffer=metric_buffer,
            extra_service=extra_service,
            metric_emitter=metric_emitter,
        )

    def run(self) -> None:
        """Run the main event loop.

        1. Validate the registry graph
        2. Wire parallel node queues
        3. Start parallel variant threads
        4. Start metrics components
        5. Start periodic stats logging
        6. Monitor parallel variant health
        """
        errors = self._registry.validate()
        if errors:
            for error in errors:
                logger.error("Graph validation error: {}", error)
            msg = f"Pipeline graph has {len(errors)} validation error(s)"
            raise ValueError(msg)

        mermaid = generate_mermaid(self._registry)
        logger.info("Pipeline graph:\n```mermaid\n{}\n```", mermaid)

        if self._config.debug.mermaid_path:
            Path(self._config.debug.mermaid_path).write_text(mermaid, encoding="utf-8")
            logger.info("Mermaid diagram written to {}", self._config.debug.mermaid_path)

        self._build_routing_table()
        self._wire_parallel_queues()
        self._start_parallel_variants()

        if self._extra_service is not None:
            self._extra_service.start()
        if self._metric_emitter is not None:
            self._metric_emitter.start()

        self._start_stats_timer()

        logger.info("Application main loop started (all nodes are parallel)")

        while not self._shutdown.is_set():
            self._check_parallel_health()
            time.sleep(0.01)

    def shutdown(self) -> None:
        """Signal all components to stop and wait for parallel variants to finish."""
        logger.info("Application shutdown requested")
        self._shutdown.set()
        self._cancel_stats_timer()

        if self._extra_service is not None:
            self._extra_service.stop()
        if self._metric_emitter is not None:
            self._metric_emitter.stop()

        self._stop_parallel_variants()

    @runtime_checkable
    class _HasErrorMessage(Protocol):
        @property
        def error_message(self) -> str: ...

    def _log_error_event(self, event: object) -> None:
        """Log error events in the main thread and optionally stop on error."""
        if isinstance(event, ConnectorError):
            logger.error("Connector error on {}: {}", event.stream_name, event.error_message)
        elif isinstance(event, self._HasErrorMessage):
            logger.error("Pipeline error: {}: {}", type(event).__name__, event.error_message)
        else:
            return

        self._stats_errors += 1

        if self._config.debug.stop_on_error:
            logger.warning("debug.stop_on_error=true -- shutting down after error")
            self.shutdown()

    def _start_stats_timer(self) -> None:
        """Start a repeating timer that logs stats periodically."""
        self._stats_interval = self._config.metrics.stats_log_interval_s
        self._stats_timer = threading.Timer(self._stats_interval, self._stats_tick)
        self._stats_timer.daemon = True
        self._stats_timer.start()

    def _stats_tick(self) -> None:
        """Execute one stats cycle and schedule the next tick."""
        if self._shutdown.is_set():
            return
        self._log_stats()
        self._stats_timer = threading.Timer(self._stats_interval, self._stats_tick)
        self._stats_timer.daemon = True
        self._stats_timer.start()

    def _cancel_stats_timer(self) -> None:
        """Cancel the periodic stats timer."""
        if self._stats_timer is not None:
            self._stats_timer.cancel()
            self._stats_timer = None

    def _log_stats(self) -> None:
        """Log pipeline stats summary."""
        per_node_summary = ", ".join(f"{k}={v}" for k, v in sorted(self._stats_per_node.items()))
        logger.info(
            "[STATS] processed={} in_progress={} errors={} | {}",
            self._stats_processed,
            self._stats_in_progress,
            self._stats_errors,
            per_node_summary,
        )

    def _build_routing_table(self) -> None:
        """Precompute producer -> consumer routing from the validated graph.

        Uses exact type matching. Routing table stores (consumer_name,
        expected_input_type) pairs. At runtime, ``_route_to_consumers``
        checks ``type(result) is expected_type`` to filter matching consumers.
        """
        self._routing_table = self._registry.compute_routing_table()
        for name, pairs in self._routing_table.items():
            consumer_names = [cn for cn, _ in pairs]
            logger.debug("Routing table: {} -> {}", name, consumer_names)

    def _wire_parallel_queues(self) -> None:
        """Pre-create input queues for all non-source nodes (parallel and inline sinks)."""
        for name, node in self._registry.nodes.items():
            if node.variant not in SOURCE_VARIANTS:
                queue_name = f"{name}:input"
                self._queue_bus.get_queue(queue_name)
                logger.debug("Wired input queue for node {}", name)

    def _start_parallel_variants(self) -> None:
        """Start all nodes in dedicated threads."""
        for name, node in self._registry.nodes.items():
            thread = threading.Thread(
                target=self._run_parallel_node,
                args=(node,),
                name=f"parallel-{name}",
                daemon=True,
            )
            self._parallel_threads[name] = thread
            thread.start()
            logger.info("Started parallel variant thread for {}", name)

    def _run_parallel_node(self, node: GraphNode) -> None:
        """Run a parallel variant node in a loop until shutdown.

        Handles three categories:
        - Source variants: produce events, route to consumer input queues
        - Processor variants: read from input queue, transform, route outputs
        - Sink variants: read from input queue, consume (side effects)

        If a source or processor returns a list or tuple, each element
        is routed individually to downstream consumers.
        """
        while not self._shutdown.is_set():
            try:
                if node.variant in SOURCE_VARIANTS:
                    result = cast("object", node.instance())
                    if result is not None:
                        self._route_result(node, result)
                elif node.variant in SINK_VARIANTS:
                    event = self._read_from_input_queue(node, timeout=0.1)
                    if event is not None:
                        node.instance(event)
                else:
                    event = self._read_from_input_queue(node, timeout=0.1)
                    if event is not None:
                        result = cast("object", node.instance(event))
                        if result is not None:
                            self._route_result(node, result)
            except (RuntimeError, ValueError, TypeError, OSError, LookupError, queue.Full):
                logger.exception("Error in parallel node {}", node.name)
                if not self._shutdown.is_set():
                    time.sleep(0.1)

    def _route_result(self, node: GraphNode, result: object) -> None:
        """Route a result, unpacking list/tuple containers into individual events."""
        if isinstance(result, (list, tuple)):
            for item in result:
                self._route_to_consumers(node, item)
        else:
            self._route_to_consumers(node, result)

    def _route_to_consumers(self, node: GraphNode, result: object) -> None:
        """Route a result to downstream consumer input queues via runtime type check.

        For each (consumer_name, expected_type) in the routing table,
        checks ``isinstance(result, expected_type)`` which handles both
        exact matches and subclass relationships. Falls back to
        ``_pydantic_generic_match`` for Pydantic generic thread-safety.
        """
        consumer_pairs = self._routing_table.get(node.name, [])
        if not consumer_pairs:
            logger.debug(
                "No consumer found for {} from node {}",
                type(result).__name__,
                node.name,
            )
            return
        result_type = type(result)
        routed = False
        for consumer_name, expected_type in consumer_pairs:
            if isinstance(result, expected_type) or _pydantic_generic_match(result_type, expected_type):
                self._queue_bus.put(f"{consumer_name}:input", result)
                routed = True
        if not routed:
            logger.debug(
                "No type match for {} from node {} (expected: {})",
                result_type.__name__,
                node.name,
                [t.__name__ for _, t in consumer_pairs],
            )

    def _read_from_input_queue(self, node: GraphNode, timeout: float) -> object | None:
        """Read one event from a parallel node's input queue."""
        queue_name = f"{node.name}:input"
        try:
            result: object = self._queue_bus.get(queue_name, timeout=timeout)
        except queue.Empty:
            return None
        else:
            return result

    def _stop_parallel_variants(self) -> None:
        """Signal stop and join all parallel variant threads."""
        for node in self._registry.nodes.values():
            stop_event = self._get_stop_event(node.instance)
            if stop_event is not None:
                stop_event.set()

        for name, thread in self._parallel_threads.items():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("Parallel variant thread {} did not stop within timeout", name)

    def _check_parallel_health(self) -> None:
        """Check if parallel variant threads are still alive, restart if crashed."""
        for name, thread in list(self._parallel_threads.items()):
            if not thread.is_alive() and not self._shutdown.is_set():
                node = self._registry.get_node(name)
                if node is not None:
                    self._restart_counts[name] = self._restart_counts.get(name, 0) + 1
                    count = self._restart_counts[name]
                    logger.warning("Restarting parallel variant {} (restart #{})", name, count)

                    new_thread = threading.Thread(
                        target=self._run_parallel_node,
                        args=(node,),
                        name=f"parallel-{name}",
                        daemon=True,
                    )
                    self._parallel_threads[name] = new_thread
                    new_thread.start()

    def _get_stop_event(self, instance: object) -> threading.Event | None:
        """Retrieve the stop event from a parallel variant using protocol isinstance checks."""
        if isinstance(
            instance,
            (ThreadSource, ProcessSource, ThreadProcessor, ThreadPoolProcessor, ProcessPoolProcessor, ThreadSink),
        ):
            return instance.stop
        return None


def _node_name(node: object) -> str:
    """Extract a node name from the ``name`` attribute or the class name."""
    if isinstance(node, _HasName):
        return node.name or type(node).__name__
    return type(node).__name__


@runtime_checkable
class _HasName(Protocol):
    """Structural check for nodes with a ``name`` attribute."""

    @property
    def name(self) -> str: ...
