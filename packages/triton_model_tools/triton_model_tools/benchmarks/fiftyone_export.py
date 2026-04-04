"""Optional FiftyOne integration for visual inspection of detection results.

FiftyOne is an optional dependency. Import errors are raised with a clear
message directing the user to install fiftyone.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
from loguru import logger

from triton_model_tools.models.detection import Detection, ImageDetectionResult

FIFTYONE_INSTALL_MSG = "fiftyone is required for visual inspection. Install it with: pip install fiftyone"


@lru_cache(maxsize=4096)
def _read_image_dimensions(filepath: str) -> tuple[int, int]:
    """Read image width and height from disk. Results are cached by filepath.

    Returns:
        Tuple of (width, height) in pixels.

    Raises:
        FileNotFoundError: If the image cannot be read.

    """
    img = cv2.imread(filepath)
    if img is None:
        msg = f"Cannot read image to determine dimensions: {filepath}"
        raise FileNotFoundError(msg)
    h: int = img.shape[0]
    w: int = img.shape[1]
    return w, h


def _detection_to_fiftyone_bbox(
    det: Detection,
    img_w: int,
    img_h: int,
) -> list[float]:
    """Convert absolute-pixel Detection coords to FiftyOne normalized [x, y, w, h]."""
    return [
        det.x1 / img_w,
        det.y1 / img_h,
        (det.x2 - det.x1) / img_w,
        (det.y2 - det.y1) / img_h,
    ]


def _build_fo_detections(
    img_result: ImageDetectionResult,
    img_w: int,
    img_h: int,
    fo: Any,  # noqa: ANN401
    *,
    det_field: str = "detections",
) -> tuple[list[Any], str]:
    """Build a list of fo.Detection objects for one image.

    Returns:
        Tuple of (detection_list, det_field) for assignment on the sample.

    """
    detections_list: list[Any] = []
    for det in img_result.detections:
        fo_det: Any = fo.Detection(
            label=det.class_name or str(det.class_id),
            bounding_box=_detection_to_fiftyone_bbox(det, img_w, img_h),
            confidence=det.confidence,
        )
        detections_list.append(fo_det)
    return detections_list, det_field


def export_to_fiftyone(
    results: list[ImageDetectionResult],
    dataset_name: str,
) -> None:
    """Export detection results to a FiftyOne dataset.

    Args:
        results: List of per-image detection results to export.
        dataset_name: Name for the FiftyOne dataset.

    Raises:
        ImportError: If fiftyone is not installed.
        FileNotFoundError: If an image file cannot be read for dimensions.

    """
    try:
        import fiftyone as fo
    except ImportError as exc:
        raise ImportError(FIFTYONE_INSTALL_MSG) from exc

    dataset: Any = fo.Dataset(name=dataset_name, overwrite=True)

    samples: list[Any] = []
    for img_result in results:
        filepath = img_result.image_path
        sample: Any = fo.Sample(filepath=filepath)

        img_w, img_h = _read_image_dimensions(filepath)
        detections_list, _ = _build_fo_detections(img_result, img_w, img_h, fo)
        sample.detections = fo.Detections(detections=detections_list)
        samples.append(sample)

    dataset.add_samples(samples)
    logger.info("Created FiftyOne dataset {!r} with {} samples", dataset_name, len(samples))


def export_comparison_to_fiftyone(
    local_results: list[ImageDetectionResult],
    triton_results: list[ImageDetectionResult],
    dataset_name: str,
) -> None:
    """Export two sets of detection results into one FiftyOne dataset for visual drift comparison.

    Each image gets two detection fields: ``local_detections`` and ``triton_detections``.

    Args:
        local_results: Detection results from local inference (e.g., ONNX).
        triton_results: Detection results from Triton inference.
        dataset_name: Name for the FiftyOne dataset.

    Raises:
        ImportError: If fiftyone is not installed.
        FileNotFoundError: If an image file cannot be read for dimensions.
        ValueError: If the result lists have mismatched images.

    """
    try:
        import fiftyone as fo
    except ImportError as exc:
        raise ImportError(FIFTYONE_INSTALL_MSG) from exc

    local_by_stem = {Path(r.image_path).stem: r for r in local_results}
    triton_by_stem = {Path(r.image_path).stem: r for r in triton_results}

    common_stems = sorted(local_by_stem.keys() & triton_by_stem.keys())
    if not common_stems:
        msg = "No matching image stems between local and Triton results"
        raise ValueError(msg)

    local_only = local_by_stem.keys() - triton_by_stem.keys()
    triton_only = triton_by_stem.keys() - local_by_stem.keys()
    if local_only:
        logger.warning("Images only in local results (skipped): {}", sorted(local_only))
    if triton_only:
        logger.warning("Images only in Triton results (skipped): {}", sorted(triton_only))

    dataset: Any = fo.Dataset(name=dataset_name, overwrite=True)
    samples: list[Any] = []

    for stem in common_stems:
        local_r = local_by_stem[stem]
        triton_r = triton_by_stem[stem]

        filepath = local_r.image_path
        sample: Any = fo.Sample(filepath=filepath)

        img_w, img_h = _read_image_dimensions(filepath)

        local_dets, _ = _build_fo_detections(local_r, img_w, img_h, fo, det_field="local_detections")
        triton_dets, _ = _build_fo_detections(triton_r, img_w, img_h, fo, det_field="triton_detections")

        sample["local_detections"] = fo.Detections(detections=local_dets)
        sample["triton_detections"] = fo.Detections(detections=triton_dets)
        samples.append(sample)

    dataset.add_samples(samples)
    logger.info(
        "Created comparison dataset {!r} with {} samples ({} local-only skipped, {} triton-only skipped)",
        dataset_name,
        len(samples),
        len(local_only),
        len(triton_only),
    )


def launch_fiftyone_app(
    results: list[ImageDetectionResult],
    dataset_name: str,
) -> None:
    """Export results and launch the FiftyOne interactive app.

    Args:
        results: List of per-image detection results to visualize.
        dataset_name: Name for the FiftyOne dataset.

    Raises:
        ImportError: If fiftyone is not installed.

    """
    try:
        import fiftyone as fo
    except ImportError as exc:
        raise ImportError(FIFTYONE_INSTALL_MSG) from exc

    export_to_fiftyone(results, dataset_name)
    dataset: Any = fo.load_dataset(dataset_name)
    session: Any = fo.launch_app(dataset)
    session.wait()
