"""Shared statistics computation for benchmark modules."""

import numpy as np

from triton_model_tools.models.constants import MILLISECONDS_PER_SECOND

PERCENTILE_50 = 50
PERCENTILE_95 = 95
PERCENTILE_99 = 99


def compute_statistics(
    latencies_ms: list[float],
) -> tuple[float, float, float, float, float, float]:
    """Compute benchmark statistics from a list of per-item latencies in milliseconds.

    Returns:
        Tuple of (total_time_s, avg_ms, throughput_fps, p50_ms, p95_ms, p99_ms).

    """
    arr = np.array(latencies_ms, dtype=np.float64)
    total_s = float(np.sum(arr)) / MILLISECONDS_PER_SECOND
    avg_ms = float(np.mean(arr))
    throughput = len(latencies_ms) / total_s if total_s > 0 else 0.0
    p50 = float(np.percentile(arr, PERCENTILE_50))
    p95 = float(np.percentile(arr, PERCENTILE_95))
    p99 = float(np.percentile(arr, PERCENTILE_99))
    return total_s, avg_ms, throughput, p50, p95, p99
