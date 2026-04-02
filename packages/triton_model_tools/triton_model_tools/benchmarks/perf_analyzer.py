"""Integration with Triton's perf_analyzer CLI tool."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

PERF_ANALYZER_COMMAND = "perf_analyzer"
DEFAULT_TIMEOUT_SECONDS = 300
FIRST_DATA_ROW_INDEX = 0


def is_perf_analyzer_available() -> bool:
    """Check whether the perf_analyzer binary is on PATH."""
    return shutil.which(PERF_ANALYZER_COMMAND) is not None


class PerfAnalyzerConfig(BaseModel):
    """Configuration for perf_analyzer run."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    protocol: Literal["grpc", "http"] = "grpc"
    url: str = "localhost:8001"
    concurrency: int = Field(default=1, ge=1)
    measurement_interval_ms: int = Field(default=5000, ge=1000)
    num_measurements: int = Field(default=10, ge=1)
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1)


class PerfAnalyzerResult(BaseModel):
    """Parsed results from perf_analyzer CSV output."""

    model_config = ConfigDict(frozen=True)

    avg_latency_ms: float = Field(ge=0.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    p99_latency_ms: float = Field(ge=0.0)
    throughput_infer_per_sec: float = Field(ge=0.0)
    concurrency: int = Field(ge=1)


USEC_TO_MS = 0.001


def _parse_perf_analyzer_csv(csv_output: str) -> PerfAnalyzerResult:
    reader = csv.DictReader(io.StringIO(csv_output))
    rows = list(reader)

    if not rows:
        msg = "perf_analyzer CSV output contains no data rows"
        raise ValueError(msg)

    row = rows[FIRST_DATA_ROW_INDEX]

    concurrency_key = "Concurrency"
    throughput_key = "Inferences/Second"
    avg_latency_key = "Client Send"
    p50_key = "p50 latency"
    p95_key = "p95 latency"
    p99_key = "p99 latency"

    available_keys = set(row.keys())
    for key in [throughput_key, p50_key, p95_key, p99_key]:
        if key not in available_keys:
            msg = f"Missing expected column in perf_analyzer CSV: {key}"
            raise ValueError(msg)

    concurrency_val = int(row[concurrency_key]) if concurrency_key in available_keys else 1
    throughput = float(row[throughput_key])

    avg_latency_usec = float(row.get(avg_latency_key, "0"))
    p50_usec = float(row[p50_key])
    p95_usec = float(row[p95_key])
    p99_usec = float(row[p99_key])

    return PerfAnalyzerResult(
        avg_latency_ms=avg_latency_usec * USEC_TO_MS,
        p50_latency_ms=p50_usec * USEC_TO_MS,
        p95_latency_ms=p95_usec * USEC_TO_MS,
        p99_latency_ms=p99_usec * USEC_TO_MS,
        throughput_infer_per_sec=throughput,
        concurrency=concurrency_val,
    )


def _build_perf_analyzer_command(config: PerfAnalyzerConfig) -> list[str]:
    return [
        PERF_ANALYZER_COMMAND,
        "-m",
        config.model_name,
        "-i",
        config.protocol,
        "-u",
        config.url,
        "--concurrency-range",
        str(config.concurrency),
        "--measurement-interval",
        str(config.measurement_interval_ms),
        "--measurement-request-count",
        str(config.num_measurements),
        "--csv",
    ]


def run_perf_analyzer(config: PerfAnalyzerConfig) -> PerfAnalyzerResult:
    """Run perf_analyzer and parse results.

    Args:
        config: perf_analyzer configuration.

    Returns:
        Parsed performance metrics.

    Raises:
        FileNotFoundError: If perf_analyzer is not found in PATH.
        subprocess.TimeoutExpired: If perf_analyzer exceeds timeout.
        subprocess.CalledProcessError: If perf_analyzer returns non-zero exit code.
        ValueError: If CSV output cannot be parsed.

    """
    if not is_perf_analyzer_available():
        msg = (
            f"{PERF_ANALYZER_COMMAND!r} binary not found on PATH. "
            "Install the Triton SDK or run inside the Triton SDK container "
            "(nvcr.io/nvidia/tritonserver:*-py3-sdk)."
        )
        raise FileNotFoundError(msg)

    cmd = _build_perf_analyzer_command(config)
    logger.info("Running perf_analyzer: {}", " ".join(cmd))

    completed = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=config.timeout_seconds,
        check=True,
    )

    logger.debug("perf_analyzer stdout length: {} chars", len(completed.stdout))
    if completed.stderr:
        logger.debug("perf_analyzer stderr: {}", completed.stderr[:500])

    return _parse_perf_analyzer_csv(completed.stdout)
