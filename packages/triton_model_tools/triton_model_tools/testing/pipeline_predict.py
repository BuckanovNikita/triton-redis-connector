"""Run image folder through Redis pipeline and save detection results as txt files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from triton_model_tools.inference.detection_io import save_detections_to_txt
from triton_model_tools.testing.e2e_test import ConnectorE2ETestHarness, E2ETestConfig

if TYPE_CHECKING:
    from pathlib import Path

    from triton_model_tools.models.detection import FolderInferenceResult


def run_pipeline_predict(
    image_dir: Path,
    output_dir: Path,
    config: E2ETestConfig,
) -> FolderInferenceResult:
    """Push images through Redis pipeline, collect results, save as txt files.

    Args:
        image_dir: Directory containing images to process.
        output_dir: Directory where txt result files will be written.
        config: E2E test harness configuration (Redis URL, streams, model, etc.).

    Returns:
        FolderInferenceResult with all detection results and timing.

    """
    logger.info(
        "Pipeline predict: {} -> Redis({}) -> {}",
        image_dir,
        config.input_stream,
        output_dir,
    )

    harness = ConnectorE2ETestHarness(config)
    result = harness.run_through_connector(image_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for img_result in result.results:
        txt_path = output_dir / f"{img_result.image_path}.txt"
        save_detections_to_txt(img_result.detections, txt_path)

    logger.info(
        "Pipeline predict complete: {} images, {:.2f}s total, results saved to {}",
        len(result.results),
        result.total_time_s,
        output_dir,
    )

    return result
