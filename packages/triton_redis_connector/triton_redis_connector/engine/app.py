"""Application skeleton — main entry point for the pipeline.

The Application class owns the main loop, dispatching inline nodes
and monitoring parallel variant health. Resolved from the dishka
container via EngineProvider.
"""

from __future__ import annotations

import queue
import threading
import time
import typing
from typing import TYPE_CHECKING, cast

from loguru import logger

from triton_redis_connector.engine.context import EventContext, reset_context, set_context
from triton_redis_connector.engine.diagram import generate_mermaid
from triton_redis_connector.engine.registry import INLINE_VARIANTS, SINK_VARIANTS, SOURCE_VARIANTS
from triton_redis_connector.errors.models import ConnectorError, InferenceError, PreprocessingError
from triton_redis_connector.protocols import (
    Envelope,
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
    from triton_redis_connector.config.schema import AppConfig
    from triton_redis_connector.engine.queue_bus import QueueBus
    from triton_redis_connector.engine.registry import GraphNode, PipelineRegistry


class Application:
    """Main pipeline application.

    Dispatches inline nodes in topological order, monitors parallel
    variant health, and handles graceful shutdown.
    """

    def __init__(  # noqa: D107, PLR0913
        self,
        config: AppConfig,
        registry: PipelineRegistry,
        queue_bus: QueueBus,
        extra_nodes: list[tuple[str, object]] | None = None,
        metric_buffer: MetricRecorder | None = None,
        triton_collector: LifecycleService | None = None,
        metric_emitter: LifecycleService | None = None,
    ) -> None:
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
        self._triton_collector = triton_collector
        self._metric_emitter = metric_emitter
        self._stats_processed: int = 0
        self._stats_in_progress: int = 0
        self._stats_errors: int = 0
        self._stats_per_node: dict[str, int] = {}
        self._stats_sink_writes: dict[str, int] = {}
        self._stats_timer: threading.Timer | None = None
        self._stats_interval: float = 0.0

    def run(self) -> None:
        """Run the main event loop.

        1. Validate the registry graph
        2. Wire parallel node queues
        3. Start parallel variant threads
        4. Start metrics components
        5. Start periodic stats logging
        6. Run inline dispatch loop
        7. Monitor parallel variant health
        """
        errors = self._registry.validate()
        if errors:
            for error in errors:
                logger.error("Graph validation error: {}", error)
            msg = f"Pipeline graph has {len(errors)} validation error(s)"
            raise ValueError(msg)

        mermaid = generate_mermaid(self._registry)
        logger.info("Pipeline graph:\n```mermaid\n{}\n```", mermaid)

        dispatch_order = self._registry.get_dispatch_order()
        self._wire_parallel_queues()
        self._start_parallel_variants()

        if self._triton_collector is not None:
            self._triton_collector.start()
        if self._metric_emitter is not None:
            self._metric_emitter.start()

        self._start_stats_timer()

        logger.info("Application main loop started with {} inline nodes", len(dispatch_order))

        while not self._shutdown.is_set():
            dispatched_any = self._dispatch_cycle(dispatch_order)
            self._check_parallel_health()

            if not dispatched_any:
                time.sleep(0.001)

    def shutdown(self) -> None:
        """Signal all components to stop and wait for parallel variants to finish."""
        logger.info("Application shutdown requested")
        self._shutdown.set()
        self._cancel_stats_timer()

        if self._triton_collector is not None:
            self._triton_collector.stop()
        if self._metric_emitter is not None:
            self._metric_emitter.stop()

        self._stop_parallel_variants()

    def _dispatch_cycle(self, dispatch_order: list[GraphNode]) -> bool:
        """Run one cycle of inline dispatch, returning True if any event was dispatched."""
        dispatched_any = False
        for node in dispatch_order:
            if self._shutdown.is_set():
                break
            if node.variant in SOURCE_VARIANTS:
                event: object = node.instance()
                if event is not None:
                    dispatched_any = True
                    envelope: Envelope[object] = event if isinstance(event, Envelope) else Envelope(payload=event)
                    envelope.track_state("queued")
                    self._route_to_parallel_consumers(event)
                    self._route_event(envelope.payload, dispatch_order, envelope=envelope)

        if self._drain_inline_queues(dispatch_order):
            dispatched_any = True

        return dispatched_any

    def _drain_inline_queues(self, dispatch_order: list[GraphNode]) -> bool:
        """Process events queued for inline consumer nodes by parallel producers."""
        drained_any = False
        for node in dispatch_order:
            if node.variant in SOURCE_VARIANTS:
                continue
            queue_name = f"{node.name}:input"
            while not self._shutdown.is_set():
                try:
                    event = self._queue_bus.get_nowait(queue_name)
                except queue.Empty:
                    break
                node.instance(event)
                self._stats_per_node[node.name] = self._stats_per_node.get(node.name, 0) + 1
                if node.variant in SINK_VARIANTS:
                    self._stats_processed += 1
                    self._stats_sink_writes[node.name] = self._stats_sink_writes.get(node.name, 0) + 1
                drained_any = True
        return drained_any

    def _route_event(
        self,
        event: object,
        dispatch_order: list[GraphNode],
        envelope: Envelope[object] | None = None,
    ) -> None:
        """Route an event through inline processors/sinks and parallel consumer queues."""
        self._log_error_event(event)

        if envelope is None:
            envelope = event if isinstance(event, Envelope) else Envelope(payload=event)

        token = set_context(
            EventContext(
                message_id=envelope.message_id,
                source_stream=envelope.source_stream,
                created_at=envelope.created_at,
                original_fields=dict(envelope.original_fields),
            )
        )

        try:
            for node in dispatch_order:
                if node.variant in SOURCE_VARIANTS:
                    continue
                if any(self._type_matches(event, t) for t in node.input_types):
                    envelope.track_state("processing_start")
                    self._stats_in_progress += 1
                    result: object = node.instance(event)
                    self._stats_in_progress -= 1
                    self._stats_per_node[node.name] = self._stats_per_node.get(node.name, 0) + 1

                    if node.variant in SINK_VARIANTS:
                        envelope.track_state("output_sent")
                        self._stats_processed += 1
                        self._stats_sink_writes[node.name] = self._stats_sink_writes.get(node.name, 0) + 1

                    if result is not None and node.variant not in SINK_VARIANTS:
                        child_envelope = envelope.derive(result)
                        self._route_event(result, dispatch_order, envelope=child_envelope)

        finally:
            reset_context(token)

    def _route_to_parallel_consumers(self, event: object) -> None:
        """Put an event onto input queues of matching parallel consumer nodes."""
        for consumed_type, consumer_names in self._registry.type_consumers.items():
            if not self._type_matches(event, consumed_type):
                continue
            for consumer_name in consumer_names:
                consumer_node = self._registry.get_node(consumer_name)
                if consumer_node is not None and consumer_node.variant not in INLINE_VARIANTS:
                    queue_name = f"{consumer_name}:input"
                    self._queue_bus.put(queue_name, event)

    def _log_error_event(self, event: object) -> None:
        """Log error events in the main thread and optionally stop on error."""
        if isinstance(event, ConnectorError):
            logger.error("Connector error on {}: {}", event.stream_name, event.error_message)
        elif isinstance(event, InferenceError):
            logger.error(
                "Inference error on {} v{}: {} (retry {})",
                event.model_name,
                event.model_version,
                event.error_message,
                event.retry_count,
            )
        elif isinstance(event, PreprocessingError):
            logger.error("Preprocessing error: {} ({})", event.error_message, event.decoder_type)
        else:
            return

        self._stats_errors += 1

        if self._config.debug.stop_on_error:
            logger.warning("debug.stop_on_error=true — shutting down after error")
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

    def _wire_parallel_queues(self) -> None:
        """Pre-create input queues for all non-source nodes (parallel and inline sinks)."""
        for name, node in self._registry.nodes.items():
            if node.variant not in SOURCE_VARIANTS:
                queue_name = f"{name}:input"
                self._queue_bus.get_queue(queue_name)
                logger.debug("Wired input queue for node {}", name)

    def _start_parallel_variants(self) -> None:
        """Start all parallel variant nodes in dedicated threads."""
        for name, node in self._registry.nodes.items():
            if node.variant not in INLINE_VARIANTS:
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
        """
        while not self._shutdown.is_set():
            try:
                if node.variant in SOURCE_VARIANTS:
                    result = cast("object", node.instance())
                    if result is not None:
                        self._route_to_consumers(node, result)
                elif node.variant in SINK_VARIANTS:
                    event = self._read_from_input_queue(node, timeout=0.1)
                    if event is not None:
                        node.instance(event)
                else:
                    event = self._read_from_input_queue(node, timeout=0.1)
                    if event is not None:
                        result = cast("object", node.instance(event))
                        if result is not None:
                            self._route_to_consumers(node, result)
            except (RuntimeError, ValueError, TypeError, OSError, LookupError, queue.Full):
                logger.exception("Error in parallel node {}", node.name)
                if not self._shutdown.is_set():
                    time.sleep(0.1)

    def _route_to_consumers(self, node: GraphNode, result: object) -> None:
        """Route a result from a parallel node to downstream consumer input queues.

        Uses subtype-aware matching: if a consumer is registered for a
        supertype or protocol that the result satisfies, the result is routed.
        Both parallel and inline consumers receive events via their input queues.
        """
        routed = False
        for consumed_type, consumer_names in self._registry.type_consumers.items():
            if not self._type_matches(result, consumed_type):
                continue
            for consumer_name in consumer_names:
                consumer_node = self._registry.get_node(consumer_name)
                if consumer_node is not None:
                    queue_name = f"{consumer_name}:input"
                    self._queue_bus.put(queue_name, result)
                    routed = True

        if not routed:
            logger.debug(
                "No consumer found for {} from node {}",
                type(result).__name__,
                node.name,
            )

    @staticmethod
    def _type_matches(instance: object, target_type: type) -> bool:
        """Check if an instance matches a target type, handling generic aliases."""
        origin = typing.get_origin(target_type)
        if origin is not None:
            return isinstance(instance, origin)
        return isinstance(instance, target_type)

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
            if node.variant not in INLINE_VARIANTS:
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
