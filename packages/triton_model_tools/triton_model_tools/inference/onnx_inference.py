"""Local ONNX inference client using onnxruntime."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

_ort: Any = ort

END2END_DETECTION_COLUMNS = 6
BATCH_DIM_SIZE = 3


class OnnxInferenceClient:
    """Run inference on a local ONNX model file via onnxruntime.

    Accepts a path to an .onnx file and provides a simple infer() method
    that mirrors the TritonInferenceClient.infer_raw() return format.
    """

    def __init__(self, onnx_path: str | Path) -> None:
        """Load an ONNX model and create an inference session.

        Args:
            onnx_path: Path to the .onnx model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the file does not have .onnx extension.

        """
        resolved = Path(onnx_path).resolve()
        if not resolved.exists():
            msg = f"ONNX model not found: {resolved}"
            raise FileNotFoundError(msg)
        if resolved.suffix != ".onnx":
            msg = f"Expected .onnx file, got: {resolved.suffix}"
            raise ValueError(msg)

        session_options: Any = _ort.SessionOptions()
        session_options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session: Any = _ort.InferenceSession(
            str(resolved),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name: str = self.session.get_inputs()[0].name
        self._output_names: list[str] = [o.name for o in self.session.get_outputs()]

    @property
    def input_name(self) -> str:
        """Name of the model's input tensor."""
        return self._input_name

    @property
    def output_names(self) -> list[str]:
        """Names of all model output tensors."""
        return list(self._output_names)

    def detect_end2end(self) -> bool:
        """Heuristic: if the model has a single output shaped (1, N, 6), it is end2end."""
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            return False
        shape = outputs[0].shape
        if shape is None or len(shape) != BATCH_DIM_SIZE:
            return False
        last_dim: int = shape[-1]
        return last_dim == END2END_DETECTION_COLUMNS

    def infer(self, tensor: npt.NDArray[np.float32]) -> dict[str, npt.NDArray[Any]]:
        """Run inference on a preprocessed float32 tensor.

        Args:
            tensor: Input tensor with batch dimension (e.g. [1, 3, 640, 640]).

        Returns:
            Dict mapping output name to numpy array (float32).

        """
        raw_outputs = self.session.run(
            self._output_names,
            {self._input_name: tensor},
        )
        return {
            name: np.asarray(out, dtype=np.float32) for name, out in zip(self._output_names, raw_outputs, strict=True)
        }
