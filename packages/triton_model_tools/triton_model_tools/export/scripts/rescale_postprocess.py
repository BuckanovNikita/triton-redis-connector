import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            detections = pb_utils.get_input_tensor_by_name(request, "DETECTIONS").as_numpy()
            scale = pb_utils.get_input_tensor_by_name(request, "SCALE").as_numpy()
            pad_left = pb_utils.get_input_tensor_by_name(request, "PAD_LEFT").as_numpy()
            pad_top = pb_utils.get_input_tensor_by_name(request, "PAD_TOP").as_numpy()

            if detections.ndim == 3:
                detections = detections[0]

            rescaled = detections.copy()
            s = float(scale.flat[0])
            pl = float(pad_left.flat[0])
            pt = float(pad_top.flat[0])

            if s > 0 and len(rescaled) > 0:
                rescaled[:, 0] = (detections[:, 0] - pl) / s
                rescaled[:, 1] = (detections[:, 1] - pt) / s
                rescaled[:, 2] = (detections[:, 2] - pl) / s
                rescaled[:, 3] = (detections[:, 3] - pt) / s

            output = pb_utils.Tensor(
                "RESCALED_DETECTIONS",
                rescaled[np.newaxis, ...].astype(np.float32),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
        return responses
