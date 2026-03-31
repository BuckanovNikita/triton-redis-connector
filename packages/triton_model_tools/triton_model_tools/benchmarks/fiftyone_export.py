"""Optional FiftyOne integration for visual inspection of detection results.

FiftyOne is an optional dependency. Import errors are raised with a clear
message directing the user to install fiftyone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from triton_model_tools.models.detection import ImageDetectionResult

FIFTYONE_INSTALL_MSG = "fiftyone is required for visual inspection. Install it with: pip install fiftyone"


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

    """
    try:
        import fiftyone as fo  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(FIFTYONE_INSTALL_MSG) from exc

    dataset: Any = fo.Dataset(name=dataset_name, overwrite=True)

    samples: list[Any] = []
    for img_result in results:
        sample: Any = fo.Sample(filepath=img_result.image_path)

        detections_list: list[Any] = []
        for det in img_result.detections:
            img_w = max(det.x2, 1.0)
            img_h = max(det.y2, 1.0)
            fo_det: Any = fo.Detection(
                label=det.class_name or str(det.class_id),
                bounding_box=[
                    det.x1 / img_w,
                    det.y1 / img_h,
                    (det.x2 - det.x1) / img_w,
                    (det.y2 - det.y1) / img_h,
                ],
                confidence=det.confidence,
            )
            detections_list.append(fo_det)

        sample.detections = fo.Detections(detections=detections_list)
        samples.append(sample)

    dataset.add_samples(samples)


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
        import fiftyone as fo  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(FIFTYONE_INSTALL_MSG) from exc

    export_to_fiftyone(results, dataset_name)
    dataset: Any = fo.load_dataset(dataset_name)
    session: Any = fo.launch_app(dataset)
    session.wait()
