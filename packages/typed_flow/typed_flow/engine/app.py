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
from typed_flow.engine.metric_aggregator import extract_and_record_metrics
from typed_flow.engine.queue_bus import QueueBus
from typed_flow.engine.registry import SINK_VARIANTS, SOURCE_VARIANTS, PipelineRegistry
from typed_flow.engine.validation import validate_graph
from typed_flow.errors.models import make_processing_error
from typed_flow.metrics.buffer import MetricBuffer
from typed_flow.metrics.emitter import MetricEmitter
from typed_flow.protocols import (
    HasFlowMeta,
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
    result_origin = result_meta.get("origin")
    expected_origin = expected_meta.get("origin")
    if result_origin is None or expected_origin is None:
        return False
    return result_origin is expected_origin and result_meta.get("args") == expected_meta.get("args")


def _extract_event_age_ms(event: object) -> float | None:
    """Extract event age from Redis message ID timestamp in flow_meta.

    Redis message IDs have format ``{ms_timestamp}-{seq}``. Returns
    wall-clock age in ms, or None if not available.
    """
    if not isinstance(event, HasFlowMeta):
        return None
    msg_id = event.flow_meta.message_id
    if not msg_id:
        return None
    dash = msg_id.find("-")
    if dash <= 0:
        return None
    try:
        created_ms = int(msg_id[:dash])
    except ValueError:
        return None
    return time.time() * 1000.0 - created_ms


class _NodeStats:
    """Per-node statistics tracker with interval snapshot support.

    Thread-safe via a lock. Tracks processed count, error count,
    total processing time, and max event age per interval.
    """

    __slots__ = (
        "_errors",
        "_interval_ms",
        "_lock",
        "_max_event_age_ms",
        "_prev_errors",
        "_prev_processed",
        "_processed",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processed: int = 0
        self._errors: int = 0
        self._interval_ms: float = 0.0
        self._max_event_age_ms: float = 0.0
        self._prev_processed: int = 0
        self._prev_errors: int = 0

    def record_call(self, elapsed_ms: float, event_age_ms: float | None = None) -> None:
        with self._lock:
            self._processed += 1
            self._interval_ms += elapsed_ms
            if event_age_ms is not None and event_age_ms > self._max_event_age_ms:
                self._max_event_age_ms = event_age_ms

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def snapshot_and_reset_interval(self) -> tuple[int, int, float, float]:
        """Return (interval_processed, interval_errors, avg_ms, max_age_ms) and reset interval."""
        with self._lock:
            interval_processed = self._processed - self._prev_processed
            interval_errors = self._errors - self._prev_errors
            avg_ms = self._interval_ms / interval_processed if interval_processed > 0 else 0.0
            max_age = self._max_event_age_ms
            self._prev_processed = self._processed
            self._prev_errors = self._errors
            self._interval_ms = 0.0
            self._max_event_age_ms = 0.0
            return interval_processed, interval_errors, avg_ms, max_age


class Application:
    """Main pipeline application.

    All nodes run as parallel variants in dedicated threads. The main
    loop monitors variant health and manages graceful shutdown.
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
        self._restart_times: dict[str, float] = {}
        self._metric_buffer = metric_buffer
        self._extra_service = extra_service
        self._metric_emitter = metric_emitter
        self._node_stats: dict[str, _NodeStats] = {}
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

        for name in self._registry.nodes:
            self._node_stats[name] = _NodeStats()

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
        """Log per-node stats: processed, errors, avg processing time, max event age."""
        total_processed = 0
        total_errors = 0
        for name in sorted(self._node_stats):
            stats = self._node_stats[name]
            processed, errs, avg_ms, max_age = stats.snapshot_and_reset_interval()
            total_processed += processed
            total_errors += errs
            if processed == 0 and errs == 0:
                continue
            age_part = f" max_age={int(max_age)}ms" if max_age > 0 else ""
            logger.info(
                "[STATS] {} | processed={} errors={} avg={}ms{}",
                name,
                processed,
                errs,
                int(avg_ms),
                age_part,
            )
        full_queues = self._queue_bus.full_queues()
        if full_queues:
            logger.error("[STATS] FULL queues (backpressure): {}", full_queues)
        queue_depths = self._queue_bus.queue_depths()
        active_queues = {k: v for k, v in queue_depths.items() if v > 0 and k not in full_queues}
        if active_queues:
            logger.info("[STATS] queues: {}", active_queues)
        logger.info(
            "[STATS] total: processed={} errors={} restarts={}",
            total_processed,
            total_errors,
            sum(self._restart_counts.values()),
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

        Tracks per-node stats (processed, errors, in_progress, timing, event age).
        On processing errors, emits a ProcessingError to downstream error sinks.
        """
        recorder = self._metric_buffer
        stats = self._node_stats.get(node.name)
        while not self._shutdown.is_set():
            try:
                if node.variant in SOURCE_VARIANTS:
                    result = cast("object", node.instance())
                    if result is not None:
                        self._handle_source_result(node, result, recorder, stats)
                    else:
                        time.sleep(0.001)
                elif node.variant in SINK_VARIANTS:
                    event = self._read_from_input_queue(node, timeout=0.1)
                    if event is not None:
                        self._handle_sink_event(node, event, recorder, stats)
                else:
                    event = self._read_from_input_queue(node, timeout=0.1)
                    if event is not None:
                        self._handle_processor_event(node, event, recorder, stats)
            except Exception:  # noqa: BLE001
                logger.exception("Fatal error in parallel node {}", node.name)
                if stats is not None:
                    stats.record_error()
                if not self._shutdown.is_set():
                    time.sleep(1.0)

    def _handle_source_result(
        self,
        node: GraphNode,
        result: object,
        recorder: MetricRecorder | None,
        stats: _NodeStats | None,
    ) -> None:
        """Process a source result: track metrics, compute event age, route downstream."""
        event_age = _extract_event_age_ms(result)
        if recorder is not None:
            extract_and_record_metrics(result, recorder, node.name, 0.0)
        if stats is not None:
            stats.record_call(0.0, event_age)
        self._route_result(node, result)

    def _handle_sink_event(
        self,
        node: GraphNode,
        event: object,
        recorder: MetricRecorder | None,
        stats: _NodeStats | None,
    ) -> None:
        """Process a sink event: track timing, errors."""
        event_age = _extract_event_age_ms(event)
        t0 = time.perf_counter()
        try:
            node.instance(event)
        except Exception as exc:  # noqa: BLE001
            if stats is not None:
                stats.record_error()
            self._emit_processing_error(node, exc, event)
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if recorder is not None:
            extract_and_record_metrics(event, recorder, node.name, elapsed_ms)
        if stats is not None:
            stats.record_call(elapsed_ms, event_age)

    def _handle_processor_event(
        self,
        node: GraphNode,
        event: object,
        recorder: MetricRecorder | None,
        stats: _NodeStats | None,
    ) -> None:
        """Process a processor event: track timing, errors, route result."""
        t0 = time.perf_counter()
        try:
            result = cast("object", node.instance(event))
        except Exception as exc:  # noqa: BLE001
            if stats is not None:
                stats.record_error()
            self._emit_processing_error(node, exc, event)
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if result is not None:
            if recorder is not None:
                extract_and_record_metrics(result, recorder, node.name, elapsed_ms)
            if stats is not None:
                stats.record_call(elapsed_ms)
            self._route_result(node, result)
        elif stats is not None:
            stats.record_call(elapsed_ms)

    def _emit_processing_error(self, node: GraphNode, exc: BaseException, event: object) -> None:
        """Create a ProcessingError and route it via normal graph routing."""
        error = make_processing_error(node.name, exc, event)
        logger.warning(
            "Processing error in {}: {} - {}",
            node.name,
            type(exc).__name__,
            exc,
        )
        self._route_result(node, error)

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
        """Check if parallel variant threads are still alive, restart if crashed.

        Waits at least 1s between restarts to avoid BUSYGROUP spam
        when sources reconnect repeatedly.
        """
        now = time.monotonic()
        for name, thread in list(self._parallel_threads.items()):
            if not thread.is_alive() and not self._shutdown.is_set():
                node = self._registry.get_node(name)
                if node is None:
                    continue
                last_restart = self._restart_times.get(name, 0.0)
                if now - last_restart < 1.0:
                    continue

                self._restart_counts[name] = self._restart_counts.get(name, 0) + 1
                self._restart_times[name] = now
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
