"""DALI preprocessing pipeline and ensemble config generation for Triton."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader
from loguru import logger
from pydantic import BaseModel

from triton_model_tools.models import (
    EnsembleStep,
    EnsembleStepMapping,
    TritonDataType,
    TritonInstanceKind,
    TritonModelConfig,
    TritonTensorSpec,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    keep_trailing_newline=True,
    autoescape=False,
)
_DALI_TEMPLATE = _JINJA_ENV.get_template("dali_preprocess.py.j2")

DALI_PREPROCESS_SUFFIX = "_dali_preprocess"
DALI_ENSEMBLE_SUFFIX = "_dali_ensemble"
DALI_NMS_ENSEMBLE_SUFFIX = "_dali_nms_ensemble"
DALI_RESCALE_ENSEMBLE_SUFFIX = "_dali_rescale_ensemble"
DALI_NMS_RESCALE_ENSEMBLE_SUFFIX = "_dali_nms_rescale_ensemble"


class DaliPipelineConfig(BaseModel):
    """Configuration for DALI preprocessing pipeline generation."""

    target_size: tuple[int, int] = (640, 640)
    pad_value: int = 114
    normalize_div: float = 255.0
    output_layout: str = "CHW"


def generate_dali_pipeline_script(config: DaliPipelineConfig) -> str:
    """Generate a Python script defining a DALI pipeline for Triton's DALI backend.

    Implements correct ultralytics letterbox preprocessing:
    1. Decode image (mixed device for GPU acceleration)
    2. Aspect-preserving resize to fit within target_size
    3. Pad shorter dimension with pad_value (114 = ultralytics gray)
    4. BGR to RGB conversion
    5. Normalize by dividing by 255.0, transpose to CHW layout

    Args:
        config: DALI pipeline configuration parameters.

    Returns:
        Valid Python script string for Triton DALI backend (model.dali).

    """
    target_h, target_w = config.target_size
    return _DALI_TEMPLATE.render(
        target_h=target_h,
        target_w=target_w,
        pad_value=config.pad_value,
        normalize_div=config.normalize_div,
        output_layout=config.output_layout,
    )


def build_dali_model_config(
    model_name: str,
    config: DaliPipelineConfig,
) -> TritonModelConfig:
    """Build a TritonModelConfig for the DALI preprocessing model.

    Args:
        model_name: Base model name (e.g. "yolo11s"). The DALI model
            will be named "{model_name}_dali_preprocess".
        config: DALI pipeline configuration.

    Returns:
        TritonModelConfig with DALI backend settings.

    """
    target_h, target_w = config.target_size
    dali_name = f"{model_name}{DALI_PREPROCESS_SUFFIX}"

    return TritonModelConfig(
        name=dali_name,
        backend="dali",
        max_batch_size=0,
        inputs=[
            TritonTensorSpec(
                name="INPUT_0",
                data_type=TritonDataType.TYPE_UINT8,
                dims=[-1],
            ),
        ],
        outputs=[
            TritonTensorSpec(
                name="OUTPUT_0",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1, 3, target_h, target_w],
            ),
        ],
        instance_kind=TritonInstanceKind.KIND_GPU,
    )


def build_ensemble_config(
    model_name: str,
    preprocess_name: str,
    inference_name: str,
    inference_output_dims: list[int] | None = None,
) -> TritonModelConfig:
    """Build an ensemble model config chaining DALI preprocess and inference.

    Args:
        model_name: Name for the ensemble model (e.g. "yolo11s_dali_ensemble").
        preprocess_name: Name of the DALI preprocessing model.
        inference_name: Name of the inference model.
        inference_output_dims: Output tensor dimensions from the inference model.
            Must match the inference model's config.pbtxt output dims exactly.

    Returns:
        TritonModelConfig with ensemble platform and step mappings.

    """
    output_dims = inference_output_dims if inference_output_dims is not None else [-1]
    return TritonModelConfig(
        name=model_name,
        platform="ensemble",
        max_batch_size=0,
        inputs=[
            TritonTensorSpec(
                name="INPUT_IMAGE",
                data_type=TritonDataType.TYPE_UINT8,
                dims=[-1],
            ),
        ],
        outputs=[
            TritonTensorSpec(
                name="OUTPUT_DETECTIONS",
                data_type=TritonDataType.TYPE_FP32,
                dims=output_dims,
            ),
        ],
        ensemble_scheduling=[
            EnsembleStep(
                model_name=preprocess_name,
                model_version=-1,
                input_map=[
                    EnsembleStepMapping(key="INPUT_0", value="INPUT_IMAGE"),
                ],
                output_map=[
                    EnsembleStepMapping(key="OUTPUT_0", value="preprocessed_tensor"),
                ],
            ),
            EnsembleStep(
                model_name=inference_name,
                model_version=-1,
                input_map=[
                    EnsembleStepMapping(key="images", value="preprocessed_tensor"),
                ],
                output_map=[
                    EnsembleStepMapping(key="output0", value="OUTPUT_DETECTIONS"),
                ],
            ),
        ],
    )


def build_dali_nms_ensemble_config(
    model_name: str,
    preprocess_name: str,
    inference_name: str,
    nms_name: str,
) -> TritonModelConfig:
    """Build a 3-step ensemble: DALI preprocess -> inference -> NMS postprocess.

    Args:
        model_name: Name for the ensemble model.
        preprocess_name: Name of the DALI preprocessing model.
        inference_name: Name of the inference model.
        nms_name: Name of the NMS postprocess model.

    Returns:
        TritonModelConfig with ensemble platform and 3-step scheduling.

    """
    return TritonModelConfig(
        name=model_name,
        platform="ensemble",
        max_batch_size=0,
        inputs=[
            TritonTensorSpec(
                name="INPUT_IMAGE",
                data_type=TritonDataType.TYPE_UINT8,
                dims=[-1],
            ),
        ],
        outputs=[
            TritonTensorSpec(
                name="OUTPUT_DETECTIONS",
                data_type=TritonDataType.TYPE_FP32,
                dims=[-1],
            ),
        ],
        ensemble_scheduling=[
            EnsembleStep(
                model_name=preprocess_name,
                model_version=-1,
                input_map=[EnsembleStepMapping(key="INPUT_0", value="INPUT_IMAGE")],
                output_map=[EnsembleStepMapping(key="OUTPUT_0", value="preprocessed_tensor")],
            ),
            EnsembleStep(
                model_name=inference_name,
                model_version=-1,
                input_map=[EnsembleStepMapping(key="images", value="preprocessed_tensor")],
                output_map=[EnsembleStepMapping(key="output0", value="raw_detections")],
            ),
            EnsembleStep(
                model_name=nms_name,
                model_version=-1,
                input_map=[EnsembleStepMapping(key="INPUT_0", value="raw_detections")],
                output_map=[EnsembleStepMapping(key="OUTPUT_0", value="OUTPUT_DETECTIONS")],
            ),
        ],
    )


def build_dali_rescale_ensemble_config(
    model_name: str,
    preprocess_name: str,
    inference_name: str,
    rescale_name: str,
) -> TritonModelConfig:
    """Build a 3-step ensemble: DALI preprocess -> inference -> rescale.

    Args:
        model_name: Name for the ensemble model.
        preprocess_name: Name of the DALI preprocessing model.
        inference_name: Name of the inference model.
        rescale_name: Name of the rescale postprocess model.

    Returns:
        TritonModelConfig with ensemble platform and 3-step scheduling.

    """
    return TritonModelConfig(
        name=model_name,
        platform="ensemble",
        max_batch_size=0,
        inputs=[
            TritonTensorSpec(
                name="INPUT_IMAGE",
                data_type=TritonDataType.TYPE_UINT8,
                dims=[-1],
            ),
            TritonTensorSpec(
                name="SCALE",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1],
            ),
            TritonTensorSpec(
                name="PAD_LEFT",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1],
            ),
            TritonTensorSpec(
                name="PAD_TOP",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1],
            ),
        ],
        outputs=[
            TritonTensorSpec(
                name="OUTPUT_DETECTIONS",
                data_type=TritonDataType.TYPE_FP32,
                dims=[-1],
            ),
        ],
        ensemble_scheduling=[
            EnsembleStep(
                model_name=preprocess_name,
                model_version=-1,
                input_map=[EnsembleStepMapping(key="INPUT_0", value="INPUT_IMAGE")],
                output_map=[EnsembleStepMapping(key="OUTPUT_0", value="preprocessed_tensor")],
            ),
            EnsembleStep(
                model_name=inference_name,
                model_version=-1,
                input_map=[EnsembleStepMapping(key="images", value="preprocessed_tensor")],
                output_map=[EnsembleStepMapping(key="output0", value="raw_detections")],
            ),
            EnsembleStep(
                model_name=rescale_name,
                model_version=-1,
                input_map=[
                    EnsembleStepMapping(key="DETECTIONS", value="raw_detections"),
                    EnsembleStepMapping(key="SCALE", value="SCALE"),
                    EnsembleStepMapping(key="PAD_LEFT", value="PAD_LEFT"),
                    EnsembleStepMapping(key="PAD_TOP", value="PAD_TOP"),
                ],
                output_map=[EnsembleStepMapping(key="RESCALED_DETECTIONS", value="OUTPUT_DETECTIONS")],
            ),
        ],
    )


def build_dali_nms_rescale_ensemble_config(
    model_name: str,
    preprocess_name: str,
    inference_name: str,
    nms_name: str,
    rescale_name: str,
) -> TritonModelConfig:
    """Build a 4-step ensemble: DALI preprocess -> inference -> NMS -> rescale.

    Args:
        model_name: Name for the ensemble model.
        preprocess_name: Name of the DALI preprocessing model.
        inference_name: Name of the inference model.
        nms_name: Name of the NMS postprocess model.
        rescale_name: Name of the rescale postprocess model.

    Returns:
        TritonModelConfig with ensemble platform and 4-step scheduling.

    """
    return TritonModelConfig(
        name=model_name,
        platform="ensemble",
        max_batch_size=0,
        inputs=[
            TritonTensorSpec(
                name="INPUT_IMAGE",
                data_type=TritonDataType.TYPE_UINT8,
                dims=[-1],
            ),
            TritonTensorSpec(
                name="SCALE",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1],
            ),
            TritonTensorSpec(
                name="PAD_LEFT",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1],
            ),
            TritonTensorSpec(
                name="PAD_TOP",
                data_type=TritonDataType.TYPE_FP32,
                dims=[1],
            ),
        ],
        outputs=[
            TritonTensorSpec(
                name="OUTPUT_DETECTIONS",
                data_type=TritonDataType.TYPE_FP32,
                dims=[-1],
            ),
        ],
        ensemble_scheduling=[
            EnsembleStep(
                model_name=preprocess_name,
                model_version=-1,
                input_map=[EnsembleStepMapping(key="INPUT_0", value="INPUT_IMAGE")],
                output_map=[EnsembleStepMapping(key="OUTPUT_0", value="preprocessed_tensor")],
            ),
            EnsembleStep(
                model_name=inference_name,
                model_version=-1,
                input_map=[EnsembleStepMapping(key="images", value="preprocessed_tensor")],
                output_map=[EnsembleStepMapping(key="output0", value="raw_detections")],
            ),
            EnsembleStep(
                model_name=nms_name,
                model_version=-1,
                input_map=[EnsembleStepMapping(key="INPUT_0", value="raw_detections")],
                output_map=[EnsembleStepMapping(key="OUTPUT_0", value="nms_detections")],
            ),
            EnsembleStep(
                model_name=rescale_name,
                model_version=-1,
                input_map=[
                    EnsembleStepMapping(key="DETECTIONS", value="nms_detections"),
                    EnsembleStepMapping(key="SCALE", value="SCALE"),
                    EnsembleStepMapping(key="PAD_LEFT", value="PAD_LEFT"),
                    EnsembleStepMapping(key="PAD_TOP", value="PAD_TOP"),
                ],
                output_map=[EnsembleStepMapping(key="RESCALED_DETECTIONS", value="OUTPUT_DETECTIONS")],
            ),
        ],
    )


def _resolve_ensemble_suffix(
    nms_model_name: str | None,
    rescale_model_name: str | None,
) -> str:
    if nms_model_name and rescale_model_name:
        return DALI_NMS_RESCALE_ENSEMBLE_SUFFIX
    if rescale_model_name:
        return DALI_RESCALE_ENSEMBLE_SUFFIX
    if nms_model_name:
        return DALI_NMS_ENSEMBLE_SUFFIX
    return DALI_ENSEMBLE_SUFFIX


def _build_ensemble_for_variant(
    ensemble_name: str,
    preprocess_name: str,
    inference_name: str,
    nms_model_name: str | None,
    rescale_model_name: str | None,
) -> TritonModelConfig:
    if nms_model_name and rescale_model_name:
        return build_dali_nms_rescale_ensemble_config(
            model_name=ensemble_name,
            preprocess_name=preprocess_name,
            inference_name=inference_name,
            nms_name=nms_model_name,
            rescale_name=rescale_model_name,
        )
    if rescale_model_name:
        return build_dali_rescale_ensemble_config(
            model_name=ensemble_name,
            preprocess_name=preprocess_name,
            inference_name=inference_name,
            rescale_name=rescale_model_name,
        )
    if nms_model_name:
        return build_dali_nms_ensemble_config(
            model_name=ensemble_name,
            preprocess_name=preprocess_name,
            inference_name=inference_name,
            nms_name=nms_model_name,
        )
    return build_ensemble_config(
        model_name=ensemble_name,
        preprocess_name=preprocess_name,
        inference_name=inference_name,
    )


def build_ensemble_directory(
    repository_root: Path,
    model_name: str,
    config: DaliPipelineConfig,
    nms_model_name: str | None = None,
    rescale_model_name: str | None = None,
) -> Path:
    """Create the full Triton model repository directory for a DALI ensemble.

    Supports 2/3/4-step ensembles depending on which postprocessing models
    are provided:
    - No NMS, no rescale: DALI -> inference (2-step)
    - NMS only: DALI -> inference -> NMS (3-step)
    - Rescale only: DALI -> inference -> rescale (3-step)
    - NMS + rescale: DALI -> inference -> NMS -> rescale (4-step)

    Args:
        repository_root: Root of the Triton model repository.
        model_name: Base model name (e.g. "yolo11s").
        config: DALI pipeline configuration.
        nms_model_name: If provided, include NMS step in the ensemble.
        rescale_model_name: If provided, include rescale step in the ensemble.

    Returns:
        Path to the ensemble model directory.

    """
    preprocess_name = f"{model_name}{DALI_PREPROCESS_SUFFIX}"
    ensemble_suffix = _resolve_ensemble_suffix(nms_model_name, rescale_model_name)
    ensemble_name = f"{model_name}{ensemble_suffix}"

    preprocess_dir = repository_root / preprocess_name
    preprocess_version_dir = preprocess_dir / "1"
    preprocess_version_dir.mkdir(parents=True, exist_ok=True)

    pipeline_script = generate_dali_pipeline_script(config)
    dali_model_path = preprocess_version_dir / "dali.py"
    dali_model_path.write_text(pipeline_script)
    logger.info("Wrote DALI pipeline script to {}", dali_model_path)

    dali_config = build_dali_model_config(model_name, config)
    dali_config_path = preprocess_dir / "config.pbtxt"
    dali_config_path.write_text(dali_config.to_pbtxt())
    logger.info("Wrote DALI config to {}", dali_config_path)

    ensemble_dir = repository_root / ensemble_name
    ensemble_version_dir = ensemble_dir / "1"
    ensemble_version_dir.mkdir(parents=True, exist_ok=True)

    ensemble_config = _build_ensemble_for_variant(
        ensemble_name=ensemble_name,
        preprocess_name=preprocess_name,
        inference_name=model_name,
        nms_model_name=nms_model_name,
        rescale_model_name=rescale_model_name,
    )
    ensemble_config_path = ensemble_dir / "config.pbtxt"
    ensemble_config_path.write_text(ensemble_config.to_pbtxt())
    logger.info("Wrote ensemble config to {}", ensemble_config_path)

    return ensemble_dir


def run_dali_preprocess(
    bgr_image: npt.NDArray[np.uint8],
    config: DaliPipelineConfig,
) -> npt.NDArray[np.floating[Any]]:
    """Run preprocessing equivalent to the DALI pipeline for validation.

    Replicates the same logic as the Jinja template (dali_preprocess.py.j2)
    using OpenCV + numpy: aspect-preserving resize -> centered pad -> BGR->RGB ->
    normalize /255 -> CHW -> add batch dim.

    Args:
        bgr_image: BGR HWC uint8 numpy array.
        config: DALI pipeline configuration.

    Returns:
        Float32 tensor with shape (1, 3, target_h, target_w).

    """
    import cv2
    import numpy as np

    target_h, target_w = config.target_size
    orig_h, orig_w = bgr_image.shape[:2]

    scale = min(target_h / orig_h, target_w / orig_w)
    new_h = round(orig_h * scale)
    new_w = round(orig_w * scale)

    resized = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dh = target_h - new_h
    dw = target_w - new_w
    pad_top = round(dh / 2 - 0.1)
    pad_bottom = round(dh / 2 + 0.1)
    pad_left = round(dw / 2 - 0.1)
    pad_right = round(dw / 2 + 0.1)

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(config.pad_value, config.pad_value, config.pad_value),
    )

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / config.normalize_div
    chw = np.transpose(normalized, (2, 0, 1))
    result: npt.NDArray[np.floating[Any]] = np.expand_dims(chw, axis=0)
    return result
