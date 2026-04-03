"""Triton-specific metrics collection."""

from __future__ import annotations

from triton_redis_connector.metrics.triton_collector import TritonMetricsCollector

__all__ = [
    "TritonMetricsCollector",
]
