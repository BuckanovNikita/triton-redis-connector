"""NMS postprocessing model generation for Triton Python backend."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

from triton_model_tools.models import (
    TritonDataType,
    TritonInstanceKind,
    TritonModelConfig,
    TritonTensorSpec,
)

if TYPE_CHECKING:
    from pathlib import Path

NMS_MODEL_NAME = "nms_postprocess"


class NmsModelConfig(BaseModel):
    """Configuration for NMS postprocessing model generation."""

    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=300, ge=1)


def generate_nms_model_script(config: NmsModelConfig) -> str:
    """Generate a Python script for Triton's Python backend implementing NMS.

    The generated script processes raw YOLO detection output tensors
    through Non-Maximum Suppression using torchvision.ops.nms, with a
    pure numpy fallback if torchvision is unavailable.

    Input tensor shape: [1, num_classes+4, num_predictions] (YOLO transposed format)
    Output tensor shape: [1, N, 6] where N <= max_detections, columns = [x1, y1, x2, y2, conf, cls]

    Args:
        config: NMS configuration parameters.

    Returns:
        Valid Python script string for Triton Python backend (model.py).

    """
    return textwrap.dedent(f"""\
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

            result = np.column_stack([
                boxes_xyxy[keep],
                max_scores[keep].reshape(-1, 1),
                class_ids[keep].astype(np.float32).reshape(-1, 1),
            ])
            return result.astype(np.float32)


        class TritonPythonModel:
            def initialize(self, args):
                model_config = json.loads(args["model_config"])
                self.confidence_threshold = {config.confidence_threshold}
                self.iou_threshold = {config.iou_threshold}
                self.max_detections = {config.max_detections}

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

            def finalize(self):
                pass
    """)


def build_nms_model_config(config: NmsModelConfig) -> TritonModelConfig:
    """Build a TritonModelConfig for the NMS postprocessing model.

    Args:
        config: NMS configuration parameters.

    Returns:
        TritonModelConfig with Python backend settings for NMS.

    """
    return TritonModelConfig(
        name=NMS_MODEL_NAME,
        backend="python",
        max_batch_size=0,
        inputs=[
            TritonTensorSpec(
                name="INPUT_0",
                data_type=TritonDataType.TYPE_FP32,
                dims=[-1, -1],
            ),
        ],
        outputs=[
            TritonTensorSpec(
                name="OUTPUT_0",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1, config.max_detections, 6],
            ),
        ],
        instance_kind=TritonInstanceKind.KIND_CPU,
    )


def build_nms_model_directory(
    repository_root: Path,
    config: NmsModelConfig,
) -> Path:
    """Create the Triton model repository directory for NMS postprocessing.

    Creates the directory structure:
        nms_postprocess/1/model.py
        nms_postprocess/config.pbtxt

    Args:
        repository_root: Root of the Triton model repository.
        config: NMS configuration parameters.

    Returns:
        Path to the created NMS model directory.

    """
    model_dir = repository_root / NMS_MODEL_NAME
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    script = generate_nms_model_script(config)
    model_path = version_dir / "model.py"
    model_path.write_text(script)
    logger.info("Wrote NMS model script to {}", model_path)

    model_config = build_nms_model_config(config)
    config_path = model_dir / "config.pbtxt"
    config_path.write_text(model_config.to_pbtxt())
    logger.info("Wrote NMS config to {}", config_path)

    return model_dir
