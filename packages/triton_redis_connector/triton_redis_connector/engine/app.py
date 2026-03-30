"""Application skeleton — main entry point for the pipeline.

The Application class owns the main loop, dispatching inline nodes
and monitoring parallel variant health. Resolved from the dishka
container via EngineProvider.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from loguru import logger  # pyright: ignore[reportMissingImports]

from triton_redis_connector.engine.context import EventContext, reset_context, set_context
from triton_redis_connector.engine.registry import INLINE_VARIANTS, SOURCE_VARIANTS
from triton_redis_connector.protocols import Envelope

if TYPE_CHECKING:
    from triton_redis_connector.config.schema import AppConfig
    from triton_redis_connector.engine.queue_bus import QueueBus
    from triton_redis_connector.engine.registry import GraphNode, PipelineRegistry


class Application:
    """Main pipeline application.

    Dispatches inline nodes in topological order, monitors parallel
    variant health, and handles graceful shutdown.
    """

    def __init__(  # noqa: D107
        self,
        config: AppConfig,
        registry: PipelineRegistry,
        queue_bus: QueueBus,
    ) -> None:
        self._config = config
        self._registry = registry
        self._queue_bus = queue_bus
        self._shutdown = threading.Event()
        self._parallel_threads: dict[str, threading.Thread] = {}

    def run(self) -> None:
        """Run the main event loop.

        1. Validate the registry graph
        2. Start parallel variant threads
        3. Run inline dispatch loop
        4. Monitor parallel variant health
        """
        errors = self._registry.validate()
        if errors:
            for error in errors:
                logger.error("Graph validation error: {}", error)
            msg = f"Pipeline graph has {len(errors)} validation error(s)"
            raise ValueError(msg)

        dispatch_order = self._registry.get_dispatch_order()
        self._start_parallel_variants()

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
        self._stop_parallel_variants()

    def _dispatch_cycle(self, dispatch_order: list[GraphNode]) -> bool:
        """Run one cycle of inline dispatch, returning True if any event was dispatched."""
        dispatched_any = False
        for node in dispatch_order:
            if self._shutdown.is_set():
                break
            if node.variant in SOURCE_VARIANTS:
                event: object = node.instance()  # type: ignore[operator]  # pyright: ignore[reportUnknownMemberType]
                if event is not None:
                    dispatched_any = True
                    self._route_event(event, dispatch_order)  # pyright: ignore[reportUnknownArgumentType]
        return dispatched_any

    def _route_event(self, event: object, dispatch_order: list[GraphNode]) -> None:
        """Route an event through inline processors and sinks by type matching."""
        event_type = type(event)
        envelope: Envelope[object] = event if isinstance(event, Envelope) else Envelope(payload=event)  # pyright: ignore[reportUnknownVariableType]

        token = set_context(
            EventContext(
                message_id=envelope.message_id,
                source_stream=envelope.source_stream,
                created_at=envelope.created_at,
            )
        )

        try:
            for node in dispatch_order:
                if node.variant in SOURCE_VARIANTS:
                    continue
                if event_type in node.input_types:
                    result: object = node.instance(event)  # type: ignore[operator]  # pyright: ignore[reportUnknownMemberType]
                    if result is not None:
                        self._route_event(result, dispatch_order)  # pyright: ignore[reportUnknownArgumentType]
        finally:
            reset_context(token)

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
        """Run a parallel variant node in a loop until shutdown."""
        while not self._shutdown.is_set():
            if node.variant in SOURCE_VARIANTS:
                result = node.instance()  # type: ignore[operator]
                if result is not None:
                    for out_type in node.output_types:
                        queue_name = f"{node.name}:{out_type.__name__}"
                        self._queue_bus.put(queue_name, result)
            else:
                time.sleep(0.01)

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
                    logger.error("Parallel variant {} crashed, restarting", name)
                    new_thread = threading.Thread(
                        target=self._run_parallel_node,
                        args=(node,),
                        name=f"parallel-{name}",
                        daemon=True,
                    )
                    self._parallel_threads[name] = new_thread
                    new_thread.start()

    def _get_stop_event(self, instance: object) -> threading.Event | None:
        """Safely retrieve the stop event from a parallel variant instance."""
        try:
            stop = instance.stop  # type: ignore[attr-defined]
        except AttributeError:
            return None
        if isinstance(stop, threading.Event):
            return stop
        return None
