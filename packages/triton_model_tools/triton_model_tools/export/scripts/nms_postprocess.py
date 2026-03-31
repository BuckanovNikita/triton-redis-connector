import json

import numpy as np
import triton_python_backend_utils as pb_utils


def _xywh_to_xyxy(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _numpy_nms(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]
    return np.array(keep, dtype=np.int64)


def _apply_nms(predictions, confidence_threshold, iou_threshold, max_detections):
    try:
        import torch
        import torchvision

        use_torch = True
    except ImportError:
        use_torch = False

    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.shape[0] == 4 + predictions.shape[1] or predictions.shape[0] > predictions.shape[1]:
        predictions = predictions.T

    num_classes = predictions.shape[1] - 4
    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = max_scores > confidence_threshold
    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(max_scores) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

    if use_torch:
        boxes_t = torch.from_numpy(boxes_xyxy).float()
        scores_t = torch.from_numpy(max_scores).float()
        keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold)
        keep = keep[:max_detections].numpy()
    else:
        keep = _numpy_nms(boxes_xyxy, max_scores, iou_threshold)
        keep = keep[:max_detections]

    result = np.column_stack(
        [
            boxes_xyxy[keep],
            max_scores[keep].reshape(-1, 1),
            class_ids[keep].astype(np.float32).reshape(-1, 1),
        ]
    )
    return result.astype(np.float32)


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        self.confidence_threshold = float(params["confidence_threshold"]["string_value"])
        self.iou_threshold = float(params["iou_threshold"]["string_value"])
        self.max_detections = int(params["max_detections"]["string_value"])

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            predictions = input_tensor.as_numpy()
            detections = _apply_nms(
                predictions,
                self.confidence_threshold,
                self.iou_threshold,
                self.max_detections,
            )
            if detections.shape[0] == 0:
                detections = np.zeros((0, 6), dtype=np.float32)
            output_tensor = pb_utils.Tensor("OUTPUT_0", detections[np.newaxis, ...])
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        return responses
