"""Continuous stream populator app: pushes generator output to Redis streams."""

import signal
import threading
import time
import types
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import redis

from typed_flow.connectors.stream_writer import serialize_model
from typed_flow.protocols.generators import StreamGenerator


class StreamPopulatorConfig(BaseModel):
    """Configuration for the StreamPopulatorApp."""

    model_config = ConfigDict(frozen=True)

    rate: float = Field(default=0.0, ge=0.0)
    batch_size: int = Field(default=100, ge=1)
    maxlen: int | None = Field(default=100_000, ge=1)
    approximate_maxlen: bool = True
    stats_interval_s: float = Field(default=10.0, gt=0.0)


class _GeneratorWorker:
    """Runs a single generator in a dedicated thread, pushing to Redis via pipelined XADD."""

    def __init__(
        self,
        redis_client: "redis.Redis[bytes]",
        generator: StreamGenerator,
        stream_name: str,
        config: StreamPopulatorConfig,
        shutdown: threading.Event,
    ) -> None:
        self._redis = redis_client
        self._generator = generator
        self._stream = stream_name
        self._config = config
        self._shutdown = shutdown
        self._total_pushed = 0
        self._lock = threading.Lock()

    @property
    def total_pushed(self) -> int:
        with self._lock:
            return self._total_pushed

    @property
    def generator_name(self) -> str:
        return type(self._generator).__name__

    @property
    def stream(self) -> str:
        return self._stream

    def run(self) -> None:
        """Run the generator loop until shutdown is signalled."""
        batch_size = self._config.batch_size
        target_batch_time = batch_size / self._config.rate if self._config.rate > 0 else 0.0
        prev_stats_time = time.monotonic()
        prev_stats_count = 0

        logger.info(
            "Worker started: generator={} stream={} rate={}",
            type(self._generator).__name__,
            self._stream,
            f"{self._config.rate:.0f} events/sec" if self._config.rate > 0 else "unlimited",
        )

        iterator = iter(self._generator())

        while not self._shutdown.is_set():
            batch_start = time.monotonic()
            pipe = self._redis.pipeline(transaction=False)
            batch_count = 0

            for _ in range(batch_size):
                try:
                    item = next(iterator)
                except StopIteration:
                    iterator = iter(self._generator())
                    try:
                        item = next(iterator)
                    except StopIteration:
                        logger.warning(
                            "Generator {} yields empty iterator, stopping worker", type(self._generator).__name__
                        )
                        return

                fields = serialize_model(item)
                pipe.xadd(
                    self._stream,
                    fields,
                    maxlen=self._config.maxlen,
                    approximate=self._config.approximate_maxlen,
                )
                batch_count += 1

            if batch_count > 0:
                pipe.execute()
                with self._lock:
                    self._total_pushed += batch_count

            now = time.monotonic()
            if now - prev_stats_time >= self._config.stats_interval_s:
                current_total = self.total_pushed
                interval_events = current_total - prev_stats_count
                interval_rate = interval_events / (now - prev_stats_time) if now > prev_stats_time else 0.0
                logger.info(
                    "[{}] total={} interval_rate={:.0f} events/sec stream={}",
                    type(self._generator).__name__,
                    current_total,
                    interval_rate,
                    self._stream,
                )
                prev_stats_time = now
                prev_stats_count = current_total

            if target_batch_time > 0:
                elapsed_batch = time.monotonic() - batch_start
                sleep_time = target_batch_time - elapsed_batch
                if sleep_time > 0:
                    self._shutdown.wait(timeout=sleep_time)


class StreamPopulatorApp:
    """Continuously pushes generator output to Redis streams until stopped.

    Each generator runs in its own thread with pipelined XADD writes,
    optional rate limiting, and periodic stats logging.
    """

    def __init__(  # noqa: D107
        self,
        redis_client: "redis.Redis[bytes]",
        generators: list[StreamGenerator],
        *,
        stream_overrides: dict[str, str] | None = None,
        config: StreamPopulatorConfig | None = None,
    ) -> None:
        if not generators:
            msg = "At least one generator is required"
            raise ValueError(msg)

        self._redis = redis_client
        self._generators = generators
        self._stream_overrides = stream_overrides or {}
        self._config = config or StreamPopulatorConfig()
        self._shutdown = threading.Event()
        self._workers: list[_GeneratorWorker] = []
        self._threads: list[threading.Thread] = []

    def _resolve_stream(self, generator: StreamGenerator) -> str:
        gen_name = type(generator).__name__
        return self._stream_overrides.get(gen_name, generator.stream_name)

    def run(self) -> None:
        """Run all generator workers until SIGINT/SIGTERM."""
        original_sigint: Any = signal.getsignal(signal.SIGINT)
        original_sigterm: Any = signal.getsignal(signal.SIGTERM)

        def handle_signal(signum: int, _frame: types.FrameType | None) -> None:
            logger.info("Received {}, shutting down...", signal.Signals(signum).name)
            self._shutdown.set()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        try:
            self._start_workers()
            self._shutdown.wait()
            self._join_workers()
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

        self._log_final_stats()

    def stop(self) -> None:
        """Signal all workers to stop (for programmatic use)."""
        self._shutdown.set()

    def _start_workers(self) -> None:
        for generator in self._generators:
            stream = self._resolve_stream(generator)
            worker = _GeneratorWorker(
                redis_client=self._redis,
                generator=generator,
                stream_name=stream,
                config=self._config,
                shutdown=self._shutdown,
            )
            self._workers.append(worker)

            thread = threading.Thread(
                target=worker.run,
                name=f"populator-{type(generator).__name__}-{stream}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

        logger.info("Started {} generator worker(s)", len(self._threads))

    def _join_workers(self) -> None:
        for thread in self._threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("Worker thread {} did not stop in time", thread.name)

    def _log_final_stats(self) -> None:
        total = sum(w.total_pushed for w in self._workers)
        for worker in self._workers:
            logger.info(
                "Worker {}: {} events pushed to {}",
                worker.generator_name,
                worker.total_pushed,
                worker.stream,
            )
        logger.info("Total events pushed: {}", total)
