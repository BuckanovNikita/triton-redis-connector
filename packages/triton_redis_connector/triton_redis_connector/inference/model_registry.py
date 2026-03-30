"""Model version tracker that polls Triton for available models.

Runs polling in a background daemon thread. Each worker process should
create its own tracker with its own tritonclient instance (gRPC channels
are not fork-safe).
"""

from __future__ import annotations

import threading
from typing import Any

from loguru import logger


class ModelVersionTracker:
    """Polls Triton model repository index and tracks model versions.

    Provides thread-safe lookup of latest model version and readiness status.
    The polling thread is a daemon and will be stopped when the parent process
    exits or when stop() is called.
    """

    def __init__(  # noqa: D107
        self,
        triton_client: Any,  # noqa: ANN401 — tritonclient has no type stubs
        poll_interval_s: float = 5.0,
    ) -> None:
        self._client = triton_client
        self._poll_interval_s = poll_interval_s

        self._models: dict[str, list[str]] = {}
        self._ready_models: set[str] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background polling thread."""
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="model-version-tracker",
        )
        self._poll_thread.start()
        logger.info("Model version tracker started (poll interval: {}s)", self._poll_interval_s)

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=self._poll_interval_s + 1.0)
            self._poll_thread = None
        logger.info("Model version tracker stopped")

    def get_latest_version(self, model_name: str) -> str | None:
        """Return the latest (highest) version string for a model, or None."""
        with self._lock:
            versions = self._models.get(model_name)
            if not versions:
                return None
            return versions[-1]

    def is_model_ready(self, model_name: str) -> bool:
        """Return True if the model is loaded and has available versions."""
        with self._lock:
            return model_name in self._ready_models

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._poll_once()
            self._stop_event.wait(timeout=self._poll_interval_s)

    def _poll_once(self) -> None:
        try:
            index = self._client.get_model_repository_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to poll Triton model repository index: {}", exc)
            return

        models: dict[str, list[str]] = {}
        ready: set[str] = set()

        for model_info in index:
            name = self._extract_model_name(model_info)
            version = self._extract_model_version(model_info)
            state = self._extract_model_state(model_info)

            if name is None:
                continue

            if name not in models:
                models[name] = []
            if version:
                models[name].append(version)
            if state == "READY":
                ready.add(name)

        for versions in models.values():
            versions.sort(key=self._version_sort_key)

        with self._lock:
            self._models = models
            self._ready_models = ready

    @staticmethod
    def _extract_model_name(model_info: Any) -> str | None:  # noqa: ANN401
        """Extract model name from a repository index entry."""
        if isinstance(model_info, dict):
            raw_name = model_info.get("name")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            return str(raw_name) if raw_name else None  # pyright: ignore[reportUnknownArgumentType]
        name_attr: str | None = model_info.name or None  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        return name_attr

    @staticmethod
    def _extract_model_version(model_info: Any) -> str:  # noqa: ANN401
        """Extract model version from a repository index entry."""
        if isinstance(model_info, dict):
            raw_version = model_info.get("version", "")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            return str(raw_version)  # pyright: ignore[reportUnknownArgumentType]
        return str(model_info.version) if model_info.version else ""  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

    @staticmethod
    def _extract_model_state(model_info: Any) -> str:  # noqa: ANN401
        """Extract model state from a repository index entry."""
        if isinstance(model_info, dict):
            raw_state = model_info.get("state", "")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            return str(raw_state)  # pyright: ignore[reportUnknownArgumentType]
        return str(model_info.state) if model_info.state else ""  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

    @staticmethod
    def _version_sort_key(version: str) -> int:
        try:
            return int(version)
        except ValueError:
            return 0
