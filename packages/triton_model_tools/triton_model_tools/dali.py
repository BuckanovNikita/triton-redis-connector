"""DALI preprocessing pipeline and ensemble config generation for Triton."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

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
    from pathlib import Path

DALI_PREPROCESS_SUFFIX = "_dali_preprocess"
DALI_ENSEMBLE_SUFFIX = "_dali_ensemble"


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
    return textwrap.dedent(f"""\
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali.pipeline import pipeline_def


        @pipeline_def(batch_size=1, num_threads=4, device_id=0)
        def pipe():
            encoded = fn.external_source(device="cpu", name="INPUT_0")
            images = fn.decoders.image(encoded, device="mixed", output_type=types.BGR)

            shapes = fn.peek_image_shape(encoded)
            orig_h = fn.slice(shapes, 0, 1, axes=[0])
            orig_w = fn.slice(shapes, 1, 1, axes=[0])

            target_h = {target_h}
            target_w = {target_w}

            scale_h = target_h / fn.cast(orig_h, dtype=types.FLOAT)
            scale_w = target_w / fn.cast(orig_w, dtype=types.FLOAT)
            scale = fn.min(scale_h, scale_w)

            new_h = fn.cast(fn.cast(orig_h, dtype=types.FLOAT) * scale, dtype=types.INT32)
            new_w = fn.cast(fn.cast(orig_w, dtype=types.FLOAT) * scale, dtype=types.INT32)

            resized = fn.resize(
                images,
                resize_x=new_w,
                resize_y=new_h,
                interp_type=types.INTERP_LINEAR,
            )

            padded = fn.pad(
                resized,
                axis_names="HW",
                shape=[target_h, target_w],
                fill_value={config.pad_value},
            )

            rgb = fn.color_space_conversion(padded, image_type=types.BGR, output_type=types.RGB)

            normalized = fn.crop_mirror_normalize(
                rgb,
                mean=[0.0, 0.0, 0.0],
                std=[{config.normalize_div}, {config.normalize_div}, {config.normalize_div}],
                output_layout="{config.output_layout}",
                dtype=types.FLOAT,
            )

            return normalized
    """)


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
                dims=[3, target_h, target_w],
            ),
        ],
        instance_kind=TritonInstanceKind.KIND_GPU,
    )


def build_ensemble_config(
    model_name: str,
    preprocess_name: str,
    inference_name: str,
) -> TritonModelConfig:
    """Build an ensemble model config chaining DALI preprocess and inference.

    Args:
        model_name: Name for the ensemble model (e.g. "yolo11s_dali_ensemble").
        preprocess_name: Name of the DALI preprocessing model.
        inference_name: Name of the inference model.

    Returns:
        TritonModelConfig with ensemble platform and step mappings.

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


def build_ensemble_directory(
    repository_root: Path,
    model_name: str,
    config: DaliPipelineConfig,
) -> Path:
    """Create the full Triton model repository directory structure for a DALI ensemble.

    Creates two model directories:
    - {model_name}_dali_preprocess/1/model.dali (pipeline script)
    - {model_name}_dali_preprocess/config.pbtxt
    - {model_name}_dali_ensemble/1/ (empty, ensemble has no model file)
    - {model_name}_dali_ensemble/config.pbtxt

    Args:
        repository_root: Root of the Triton model repository.
        model_name: Base model name (e.g. "yolo11s").
        config: DALI pipeline configuration.

    Returns:
        Path to the ensemble model directory.

    """
    preprocess_name = f"{model_name}{DALI_PREPROCESS_SUFFIX}"
    ensemble_name = f"{model_name}{DALI_ENSEMBLE_SUFFIX}"

    preprocess_dir = repository_root / preprocess_name
    preprocess_version_dir = preprocess_dir / "1"
    preprocess_version_dir.mkdir(parents=True, exist_ok=True)

    pipeline_script = generate_dali_pipeline_script(config)
    dali_model_path = preprocess_version_dir / "model.dali"
    dali_model_path.write_text(pipeline_script)
    logger.info("Wrote DALI pipeline script to {}", dali_model_path)

    dali_config = build_dali_model_config(model_name, config)
    dali_config_path = preprocess_dir / "config.pbtxt"
    dali_config_path.write_text(dali_config.to_pbtxt())
    logger.info("Wrote DALI config to {}", dali_config_path)

    ensemble_dir = repository_root / ensemble_name
    ensemble_version_dir = ensemble_dir / "1"
    ensemble_version_dir.mkdir(parents=True, exist_ok=True)

    ensemble_config = build_ensemble_config(
        model_name=ensemble_name,
        preprocess_name=preprocess_name,
        inference_name=model_name,
    )
    ensemble_config_path = ensemble_dir / "config.pbtxt"
    ensemble_config_path.write_text(ensemble_config.to_pbtxt())
    logger.info("Wrote ensemble config to {}", ensemble_config_path)

    return ensemble_dir
