"""Non-Maximum Suppression via torchvision."""

import numpy as np
import numpy.typing as npt
import torch
import torchvision


def nms(
    boxes: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float32],
    iou_threshold: float,
) -> list[int]:
    """Apply Non-Maximum Suppression using torchvision.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2] bounding boxes.
        scores: (N,) array of confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        List of kept indices, ordered by descending score.

    """
    if len(scores) == 0:
        return []

    boxes_t = torch.from_numpy(boxes).float()
    scores_t = torch.from_numpy(scores).float()
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold)
    return keep.tolist()
