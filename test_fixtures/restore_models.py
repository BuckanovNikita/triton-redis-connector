"""Restore test model weights (not committed to git).

Exports yolo11s ONNX via ultralytics and creates a corrupted variant
with zeroed-out first convolution layer weights.

Run: uv run python test_fixtures/restore_models.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

FIXTURES_DIR = Path(__file__).parent
MODELS_DIR = FIXTURES_DIR / "models"
CORRUPTED_DIR = FIXTURES_DIR / "corrupted_models"

YOLO_MODEL_NAME = "yolo11s"
ONNX_FILENAME = f"{YOLO_MODEL_NAME}.onnx"
TARGET_SIZE = 640


def _export_onnx(target_path: Path) -> None:
    """Export yolo11s to ONNX format via ultralytics."""
    from ultralytics import YOLO  # noqa: PLC0415

    logger.info("Exporting {} to ONNX...", YOLO_MODEL_NAME)
    model: Any = YOLO(f"{YOLO_MODEL_NAME}.pt")
    export_result: str = model.export(format="onnx", imgsz=TARGET_SIZE, dynamic=False)
    exported = Path(export_result)

    if exported.resolve() != target_path.resolve():
        exported.rename(target_path)

    logger.info("ONNX model saved to {}", target_path)


def _create_corrupted_model(source_path: Path, target_path: Path) -> None:
    """Create corrupted ONNX model by zeroing out the first conv layer weights."""
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    logger.info("Creating corrupted model from {}", source_path)
    model = onnx.load(str(source_path))

    corrupted = False
    for initializer in model.graph.initializer:
        arr = numpy_helper.to_array(initializer)
        if arr.ndim >= 4:
            zeroed = np.zeros_like(arr)
            new_tensor = numpy_helper.from_array(zeroed, name=initializer.name)
            initializer.CopyFrom(new_tensor)
            logger.info("Zeroed weights for tensor: {} (shape {})", initializer.name, arr.shape)
            corrupted = True
            break

    if not corrupted:
        logger.warning("No 4D weight tensor found; corrupting first initializer instead")
        if model.graph.initializer:
            first = model.graph.initializer[0]
            arr = numpy_helper.to_array(first)
            zeroed = np.zeros_like(arr)
            new_tensor = numpy_helper.from_array(zeroed, name=first.name)
            first.CopyFrom(new_tensor)

    onnx.save(model, str(target_path))
    logger.info("Corrupted model saved to {}", target_path)


def restore_models() -> None:
    """Export ONNX models and create corrupted variants. Idempotent."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)

    onnx_path = MODELS_DIR / ONNX_FILENAME
    corrupted_path = CORRUPTED_DIR / ONNX_FILENAME

    if onnx_path.exists():
        logger.info("ONNX model already exists at {}, skipping export", onnx_path)
    else:
        _export_onnx(onnx_path)

    if corrupted_path.exists():
        logger.info("Corrupted model already exists at {}, skipping", corrupted_path)
    elif onnx_path.exists():
        _create_corrupted_model(onnx_path, corrupted_path)
    else:
        logger.warning("Cannot create corrupted model: source ONNX not available")

    logger.info("Model restoration complete")


if __name__ == "__main__":
    restore_models()
