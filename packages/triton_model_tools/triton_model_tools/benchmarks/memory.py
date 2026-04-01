"""Optional memory monitoring for benchmarks using pynvml and psutil."""

from __future__ import annotations

import os
import threading
import time
from typing import Self

from loguru import logger

POLL_INTERVAL_S = 0.1
BYTES_PER_MB = 1024 * 1024


def _read_peak_vram_mb(gpu_index: int = 0) -> float | None:
    try:
        import pynvml  # noqa: PLC0415
    except ImportError:
        logger.debug("pynvml not available, skipping VRAM measurement")
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_bytes: int = int(mem_info.used)
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        logger.debug("Failed to query GPU memory via NVML")
        return None
    else:
        return used_bytes / BYTES_PER_MB


def _read_current_ram_mb() -> float | None:
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        logger.debug("psutil not available, skipping RAM measurement")
        return None

    process = psutil.Process(os.getpid())
    rss_bytes: int = process.memory_info().rss
    return rss_bytes / BYTES_PER_MB


class MemoryMonitor:
    """Polls GPU VRAM and process RAM usage in a background thread.

    Use as a context manager around code whose peak memory you want to measure.
    After exiting the context, ``peak_vram_mb`` and ``peak_ram_mb`` hold the
    highest observed values (or None if the corresponding library is unavailable).
    """

    def __init__(  # noqa: D107
        self, gpu_index: int = 0, poll_interval_s: float = POLL_INTERVAL_S
    ) -> None:
        self._gpu_index = gpu_index
        self._poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_vram_mb: float | None = None
        self.peak_ram_mb: float | None = None

    def __enter__(self) -> Self:  # noqa: D105
        self._stop.clear()
        self.peak_vram_mb = _read_peak_vram_mb(self._gpu_index)
        self.peak_ram_mb = _read_current_ram_mb()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:  # noqa: D105
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            vram = _read_peak_vram_mb(self._gpu_index)
            if vram is not None and (self.peak_vram_mb is None or vram > self.peak_vram_mb):
                self.peak_vram_mb = vram

            ram = _read_current_ram_mb()
            if ram is not None and (self.peak_ram_mb is None or ram > self.peak_ram_mb):
                self.peak_ram_mb = ram

            time.sleep(self._poll_interval_s)
