"""ResNet18 ONNX export and Triton model repository builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

from triton_model_tools.export.repository import build_model_directory
from triton_model_tools.models import (
    DynamicBatchingConfig,
    TritonDataType,
    TritonModelConfig,
    TritonTensorSpec,
)

RESNET18_INPUT_SIZE = 224
RESNET18_NUM_CHANNELS = 3
RESNET18_NUM_CLASSES = 1000


def export_resnet18(
    output_dir: Path,
    *,
    model_name: str = "resnet18",
    opset_version: int = 17,
    batch_size: int = 1,
) -> Path:
    """Export pretrained ResNet18 to ONNX and create Triton repository structure.

    Requires ``torch`` and ``torchvision`` to be installed. These are NOT
    declared as package dependencies — install them manually before calling:
    ``pip install torch torchvision``

    Args:
        output_dir: Root path of the Triton model repository.
        model_name: Name for the Triton model directory.
        opset_version: ONNX opset version to use for export.
        batch_size: Maximum batch size for Triton dynamic batching.

    Returns:
        Path to the created model directory.

    Raises:
        ImportError: If torch or torchvision are not installed.

    """
    try:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        msg = "torch and torchvision are required for ResNet18 export. Install them with: pip install torch torchvision"
        raise ImportError(msg) from exc

    logger.info("Loading pretrained ResNet18 (ImageNet1K_V1)")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    dummy_input = torch.randn(1, RESNET18_NUM_CHANNELS, RESNET18_INPUT_SIZE, RESNET18_INPUT_SIZE)
    onnx_path = output_dir / model_name / "1" / "model.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting to ONNX: {}", onnx_path)
    torch.onnx.export(
        model,
        dummy_input,  # type: ignore[arg-type]
        str(onnx_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    triton_config = TritonModelConfig(
        name=model_name,
        platform="onnxruntime_onnx",
        max_batch_size=batch_size,
        inputs=[
            TritonTensorSpec(
                name="input",
                data_type=TritonDataType.TYPE_FP32,
                dims=[RESNET18_NUM_CHANNELS, RESNET18_INPUT_SIZE, RESNET18_INPUT_SIZE],
            ),
        ],
        outputs=[
            TritonTensorSpec(
                name="output",
                data_type=TritonDataType.TYPE_FP32,
                dims=[RESNET18_NUM_CLASSES],
            ),
        ],
        dynamic_batching=DynamicBatchingConfig(),
    )

    model_dir = build_model_directory(
        repository_root=output_dir,
        config=triton_config,
        model_source=onnx_path,
    )

    logger.info("ResNet18 exported to Triton repository: {}", model_dir)
    return model_dir
