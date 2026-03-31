"""I/O utilities for reading/writing detection results in txt format.

Txt format: one detection per line, fields separated by spaces:
    class_id x1 y1 x2 y2 confidence

Coordinates are absolute pixel values (not normalized).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from triton_model_tools.detection import Detection, ImageDetectionResult

if TYPE_CHECKING:
    from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})
TXT_EXTENSION = ".txt"


def save_detections_to_txt(detections: list[Detection], output_path: Path) -> None:
    """Write detections to a txt file. One line per detection: class_id x1 y1 x2 y2 confidence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{d.class_id} {d.x1:.6f} {d.y1:.6f} {d.x2:.6f} {d.y2:.6f} {d.confidence:.6f}" for d in detections]
    output_path.write_text("\n".join(lines) + "\n" if lines else "")
    logger.debug("Saved {} detections to {}", len(detections), output_path)


def load_detections_from_txt(
    txt_path: Path,
    image_path: str,
    model_name: str,
) -> ImageDetectionResult:
    """Read detections from a txt file. Returns ImageDetectionResult."""
    detections: list[Detection] = []
    text = txt_path.read_text().strip()
    if text:
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 6:
                logger.warning("Skipping malformed line in {}: {}", txt_path, line)
                continue
            detections.append(
                Detection(
                    class_id=int(parts[0]),
                    x1=float(parts[1]),
                    y1=float(parts[2]),
                    x2=float(parts[3]),
                    y2=float(parts[4]),
                    confidence=float(parts[5]),
                ),
            )
    return ImageDetectionResult(
        image_path=image_path,
        model_name=model_name,
        detections=detections,
        inference_time_ms=0.0,
    )


def load_folder_detections(
    txt_dir: Path,
    model_name: str,
) -> list[ImageDetectionResult]:
    """Load all txt detection files from a directory.

    Matches txt filenames to image names by replacing .txt with the original extension.
    """
    txt_files = sorted(txt_dir.glob(f"*{TXT_EXTENSION}"))
    if not txt_files:
        msg = f"No txt files found in {txt_dir}"
        raise FileNotFoundError(msg)

    results: list[ImageDetectionResult] = []
    for txt_path in txt_files:
        image_name = txt_path.stem
        result = load_detections_from_txt(txt_path, image_path=image_name, model_name=model_name)
        results.append(result)

    logger.info("Loaded {} detection files from {}", len(results), txt_dir)
    return results
