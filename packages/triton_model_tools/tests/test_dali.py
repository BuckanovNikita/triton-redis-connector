"""Tests for DALI preprocessing pipeline and ensemble config generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from triton_model_tools.export.dali import (
    DALI_ENSEMBLE_SUFFIX,
    DALI_PREPROCESS_SUFFIX,
    DaliPipelineConfig,
    build_dali_model_config,
    build_dali_nms_ensemble_config,
    build_ensemble_config,
    build_ensemble_directory,
)

try:
    import nvidia.dali  # noqa: F401

    _HAS_DALI = True
except ImportError:
    _HAS_DALI = False

requires_dali = pytest.mark.skipif(not _HAS_DALI, reason="nvidia-dali not installed")


class TestBuildDaliModelConfig:
    def test_returns_dali_backend(self) -> None:
        config = DaliPipelineConfig()
        model_config = build_dali_model_config("yolo11s", config)
        assert model_config.backend == "dali"

    def test_input_is_uint8_bytes(self) -> None:
        config = DaliPipelineConfig()
        model_config = build_dali_model_config("yolo11s", config)
        assert len(model_config.inputs) == 1
        assert model_config.inputs[0].data_type.value == "TYPE_UINT8"
        assert model_config.inputs[0].dims == [-1]

    def test_output_is_fp32_chw(self) -> None:
        config = DaliPipelineConfig()
        model_config = build_dali_model_config("yolo11s", config)
        assert len(model_config.outputs) == 1
        assert model_config.outputs[0].data_type.value == "TYPE_FP32"
        assert model_config.outputs[0].dims == [1, 3, 640, 640]

    def test_custom_target_size_in_output_dims(self) -> None:
        config = DaliPipelineConfig(target_size=(320, 320))
        model_config = build_dali_model_config("yolo11s", config)
        assert model_config.outputs[0].dims == [1, 3, 320, 320]


class TestBuildEnsembleConfig:
    def test_has_ensemble_scheduling(self) -> None:
        model_config = build_ensemble_config(
            model_name="yolo11s_dali_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
        )
        assert model_config.ensemble_scheduling is not None
        assert len(model_config.ensemble_scheduling) == 2

    def test_first_step_is_preprocess(self) -> None:
        model_config = build_ensemble_config(
            model_name="yolo11s_dali_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
        )
        assert model_config.ensemble_scheduling is not None
        first_step = model_config.ensemble_scheduling[0]
        assert first_step.model_name == "yolo11s_dali_preprocess"

    def test_second_step_is_inference(self) -> None:
        model_config = build_ensemble_config(
            model_name="yolo11s_dali_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
        )
        assert model_config.ensemble_scheduling is not None
        second_step = model_config.ensemble_scheduling[1]
        assert second_step.model_name == "yolo11s"

    def test_step_mappings_chain_correctly(self) -> None:
        model_config = build_ensemble_config(
            model_name="yolo11s_dali_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
        )
        assert model_config.ensemble_scheduling is not None
        preprocess_step = model_config.ensemble_scheduling[0]
        inference_step = model_config.ensemble_scheduling[1]

        preprocess_output_value = preprocess_step.output_map[0].value
        inference_input_value = inference_step.input_map[0].value
        assert preprocess_output_value == inference_input_value


class TestBuildEnsembleDirectory:
    def test_creates_preprocess_directory(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        build_ensemble_directory(tmp_path, "yolo11s", config)

        preprocess_dir = tmp_path / f"yolo11s{DALI_PREPROCESS_SUFFIX}"
        assert preprocess_dir.is_dir()
        assert (preprocess_dir / "1").is_dir()

    def test_creates_ensemble_directory(self, tmp_path: Path) -> None:
        config = DaliPipelineConfig()
        build_ensemble_directory(tmp_path, "yolo11s", config)

        ensemble_dir = tmp_path / f"yolo11s{DALI_ENSEMBLE_SUFFIX}"
        assert ensemble_dir.is_dir()
        assert (ensemble_dir / "1").is_dir()


class TestBuildDaliNmsEnsembleConfig:
    def test_has_three_steps(self) -> None:
        model_config = build_dali_nms_ensemble_config(
            model_name="yolo11s_dali_nms_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
            nms_name="nms_postprocess",
        )
        assert model_config.ensemble_scheduling is not None
        assert len(model_config.ensemble_scheduling) == 3

    def test_steps_chain_correctly(self) -> None:
        model_config = build_dali_nms_ensemble_config(
            model_name="yolo11s_dali_nms_ensemble",
            preprocess_name="yolo11s_dali_preprocess",
            inference_name="yolo11s",
            nms_name="nms_postprocess",
        )
        assert model_config.ensemble_scheduling is not None
        preprocess_step = model_config.ensemble_scheduling[0]
        inference_step = model_config.ensemble_scheduling[1]
        nms_step = model_config.ensemble_scheduling[2]

        assert preprocess_step.model_name == "yolo11s_dali_preprocess"
        assert inference_step.model_name == "yolo11s"
        assert nms_step.model_name == "nms_postprocess"

        assert preprocess_step.output_map[0].value == inference_step.input_map[0].value
        assert inference_step.output_map[0].value == nms_step.input_map[0].value


@requires_dali
class TestDaliPipelineMatchesUltralytics:
    """Verify generated DALI pipeline output matches ultralytics LetterBox.

    Runs the actual DALI pipeline (generated from template) on host GPU with
    DALI_DISABLE_NVML=1. Compares against ultralytics LetterBox using SSIM
    for structured images (gradients) and structural checks (padding position,
    shape, normalization) for real photos where resize interpolation differs
    between DALI and OpenCV.

    SSIM threshold is set for gradient images where interpolation differences
    are minimal. Real fixture images verify structural correctness only
    (content position, padding region, shape) since DALI and OpenCV bilinear
    implementations differ significantly on high-frequency content.
    """

    _SSIM_THRESHOLD = 0.90
    _FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "test_fixtures" / "images"

    @staticmethod
    def _run_dali_pipeline(
        jpeg_bytes: bytes,
        target_h: int,
        target_w: int,
    ) -> npt.NDArray[np.floating[Any]]:
        import nvidia.dali.math as dmath  # pyright: ignore[reportMissingImports]
        import nvidia.dali.types as dali_types  # pyright: ignore[reportMissingImports]
        from nvidia.dali import fn  # pyright: ignore[reportMissingImports]
        from nvidia.dali.pipeline import Pipeline  # pyright: ignore[reportMissingImports]

        pipe = Pipeline(batch_size=1, num_threads=4, device_id=0)
        with pipe:
            encoded = fn.external_source(device="cpu", name="INPUT_0")
            images = fn.decoders.image(encoded, device="cpu", output_type=dali_types.BGR)

            shapes = fn.peek_image_shape(encoded)
            orig_h = fn.slice(shapes, 0, 1, axes=[0])
            orig_w = fn.slice(shapes, 1, 1, axes=[0])

            scale_h = target_h / fn.cast(orig_h, dtype=dali_types.FLOAT)
            scale_w = target_w / fn.cast(orig_w, dtype=dali_types.FLOAT)
            scale = dmath.min(scale_h, scale_w)

            new_h = fn.cast(fn.cast(orig_h, dtype=dali_types.FLOAT) * scale, dtype=dali_types.INT32)
            new_w = fn.cast(fn.cast(orig_w, dtype=dali_types.FLOAT) * scale, dtype=dali_types.INT32)

            resized = fn.resize(
                images,
                resize_x=fn.cast(new_w, dtype=dali_types.FLOAT),
                resize_y=fn.cast(new_h, dtype=dali_types.FLOAT),
                interp_type=dali_types.INTERP_LINEAR,
            )

            paste_ratio = dmath.max(
                target_h / fn.cast(new_h, dtype=dali_types.FLOAT),
                target_w / fn.cast(new_w, dtype=dali_types.FLOAT),
            )

            padded = fn.paste(
                resized,
                paste_x=0.5,
                paste_y=0.5,
                ratio=paste_ratio,
                fill_value=114,
            )
            padded = fn.crop(
                padded,
                crop_h=target_h,
                crop_w=target_w,
                crop_pos_x=0.5,
                crop_pos_y=0.5,
                out_of_bounds_policy="pad",
                fill_values=114,
            )

            rgb = fn.color_space_conversion(padded, image_type=dali_types.BGR, output_type=dali_types.RGB)
            normalized = fn.crop_mirror_normalize(
                rgb,
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
                output_layout="CHW",
                dtype=dali_types.FLOAT,
            )
            batched = fn.expand_dims(normalized, axes=[0])
            pipe.set_outputs(batched)

        pipe.build()
        encoded_np = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        pipe.feed_input("INPUT_0", [encoded_np])
        (output,) = pipe.run()
        result: npt.NDArray[np.floating[Any]] = np.array(output[0])
        return result

    @staticmethod
    def _run_ultralytics_preprocess(
        bgr_image: npt.NDArray[np.uint8],
        target_size: tuple[int, int],
    ) -> npt.NDArray[np.floating[Any]]:
        """Exact ultralytics predict preprocessing: LetterBox -> BGR2RGB -> CHW -> /255."""
        from ultralytics.data.augment import LetterBox

        letterbox = LetterBox(new_shape=target_size, auto=False, stride=32)
        letterboxed: npt.NDArray[np.uint8] = letterbox(image=bgr_image)  # type: ignore[assignment]
        rgb = letterboxed[..., ::-1]
        chw = np.ascontiguousarray(rgb.transpose((2, 0, 1)))
        normalized = chw.astype(np.float32) / 255.0
        result: npt.NDArray[np.floating[Any]] = np.expand_dims(normalized, axis=0)
        return result

    @staticmethod
    def _compute_ssim(
        a: npt.NDArray[np.floating[Any]],
        b: npt.NDArray[np.floating[Any]],
    ) -> float:
        from skimage.metrics import structural_similarity

        hwc_a = np.transpose(a[0], (1, 2, 0))
        hwc_b = np.transpose(b[0], (1, 2, 0))
        score: float = structural_similarity(hwc_a, hwc_b, channel_axis=2, data_range=1.0)
        return score

    def _compare(
        self,
        bgr_image: npt.NDArray[np.uint8],
        jpeg_bytes: bytes,
        target_size: tuple[int, int],
    ) -> float:
        target_h, target_w = target_size
        dali_out = self._run_dali_pipeline(jpeg_bytes, target_h, target_w)
        ultra_out = self._run_ultralytics_preprocess(bgr_image, target_size)
        assert dali_out.shape == ultra_out.shape
        return self._compute_ssim(dali_out, ultra_out)

    @staticmethod
    def _make_jpeg_pair(bgr: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.uint8], bytes]:
        import cv2

        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        jpeg_bytes = buf.tobytes()
        decoded: npt.NDArray[np.uint8] = cv2.imdecode(  # type: ignore[assignment]
            np.frombuffer(jpeg_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        return decoded, jpeg_bytes

    @staticmethod
    def _make_gradient_image(h: int, w: int) -> npt.NDArray[np.uint8]:
        y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
        x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
        b = np.full((h, w), 128, dtype=np.float32)
        bgr = np.stack([b, np.broadcast_to(y, (h, w)), np.broadcast_to(x, (h, w))], axis=2)
        return bgr.astype(np.uint8)

    @staticmethod
    def _find_content_bounds(
        tensor: npt.NDArray[np.floating[Any]],
        pad_value: float = 114.0 / 255.0,
    ) -> tuple[int, int, int, int]:
        hwc = np.transpose(tensor[0], (1, 2, 0))
        non_pad = np.where(~np.all(np.isclose(hwc, pad_value, atol=0.02), axis=2))
        if len(non_pad[0]) == 0:
            return 0, 0, 0, 0
        return int(non_pad[0].min()), int(non_pad[0].max()), int(non_pad[1].min()), int(non_pad[1].max())

    def _assert_structural_match(
        self,
        bgr_image: npt.NDArray[np.uint8],
        jpeg_bytes: bytes,
        target_size: tuple[int, int],
        name: str,
    ) -> None:
        target_h, target_w = target_size
        dali_out = self._run_dali_pipeline(jpeg_bytes, target_h, target_w)
        ultra_out = self._run_ultralytics_preprocess(bgr_image, target_size)

        assert dali_out.shape == ultra_out.shape, f"{name}: shape {dali_out.shape} != {ultra_out.shape}"
        assert dali_out.min() >= 0.0, f"{name}: min {dali_out.min()} < 0"
        assert dali_out.max() <= 1.0, f"{name}: max {dali_out.max()} > 1"

        d_y0, d_y1, d_x0, d_x1 = self._find_content_bounds(dali_out)
        u_y0, u_y1, u_x0, u_x1 = self._find_content_bounds(ultra_out)
        assert abs(d_y0 - u_y0) <= 1, f"{name}: top off {abs(d_y0 - u_y0)}px"
        assert abs(d_y1 - u_y1) <= 1, f"{name}: bottom off {abs(d_y1 - u_y1)}px"
        assert abs(d_x0 - u_x0) <= 1, f"{name}: left off {abs(d_x0 - u_x0)}px"
        assert abs(d_x1 - u_x1) <= 1, f"{name}: right off {abs(d_x1 - u_x1)}px"

    def test_structural_coco8_640x640(self) -> None:
        import cv2

        coco8_dir = Path("/home/nkt/datasets/coco8/images/train")
        if not coco8_dir.is_dir():
            pytest.skip("coco8 not downloaded")
        images = sorted(coco8_dir.glob("*.jpg"))
        if not images:
            pytest.skip("no coco8 images")

        for image_path in images:
            bgr = cv2.imread(str(image_path))
            jpeg = image_path.read_bytes()
            self._assert_structural_match(bgr, jpeg, (640, 640), image_path.name)

    def test_structural_coco8_608x960(self) -> None:
        import cv2

        coco8_dir = Path("/home/nkt/datasets/coco8/images/train")
        if not coco8_dir.is_dir():
            pytest.skip("coco8 not downloaded")
        images = sorted(coco8_dir.glob("*.jpg"))
        if not images:
            pytest.skip("no coco8 images")

        for image_path in images:
            bgr = cv2.imread(str(image_path))
            jpeg = image_path.read_bytes()
            self._assert_structural_match(bgr, jpeg, (608, 960), image_path.name)

    def test_padding_values_match(self) -> None:
        bgr = self._make_gradient_image(480, 640)
        decoded, jpeg = self._make_jpeg_pair(bgr)
        dali_out = self._run_dali_pipeline(jpeg, 640, 640)
        ultra_out = self._run_ultralytics_preprocess(decoded, (640, 640))
        pad_val = 114.0 / 255.0
        dali_pad_top = dali_out[0, :, :70, :]
        ultra_pad_top = ultra_out[0, :, :70, :]
        np.testing.assert_allclose(dali_pad_top, pad_val, atol=0.01)
        np.testing.assert_allclose(ultra_pad_top, pad_val, atol=0.01)

    def test_ssim_highres_1960x1200_to_640x640(self) -> None:
        bgr = self._make_gradient_image(1200, 1960)
        decoded, jpeg = self._make_jpeg_pair(bgr)
        ssim = self._compare(decoded, jpeg, (640, 640))
        assert ssim >= self._SSIM_THRESHOLD, f"1960x1200->640x640: SSIM {ssim:.4f} < {self._SSIM_THRESHOLD}"

    def test_ssim_highres_1960x1200_to_608x960(self) -> None:
        bgr = self._make_gradient_image(1200, 1960)
        decoded, jpeg = self._make_jpeg_pair(bgr)
        ssim = self._compare(decoded, jpeg, (608, 960))
        assert ssim >= self._SSIM_THRESHOLD, f"1960x1200->608x960: SSIM {ssim:.4f} < {self._SSIM_THRESHOLD}"

    def test_ssim_landscape_gradient(self) -> None:
        bgr = self._make_gradient_image(480, 640)
        decoded, jpeg = self._make_jpeg_pair(bgr)
        ssim = self._compare(decoded, jpeg, (640, 640))
        assert ssim >= self._SSIM_THRESHOLD, f"640x480->640x640: SSIM {ssim:.4f} < {self._SSIM_THRESHOLD}"

    def test_ssim_portrait_gradient(self) -> None:
        bgr = self._make_gradient_image(800, 400)
        decoded, jpeg = self._make_jpeg_pair(bgr)
        ssim = self._compare(decoded, jpeg, (640, 640))
        assert ssim >= self._SSIM_THRESHOLD, f"400x800->640x640: SSIM {ssim:.4f} < {self._SSIM_THRESHOLD}"

    def test_normalization_range(self) -> None:
        bgr = self._make_gradient_image(480, 640)
        _, jpeg = self._make_jpeg_pair(bgr)
        dali_out = self._run_dali_pipeline(jpeg, 640, 640)
        assert dali_out.min() >= 0.0
        assert dali_out.max() <= 1.0

    def test_output_shape_square(self) -> None:
        bgr = self._make_gradient_image(480, 640)
        _, jpeg = self._make_jpeg_pair(bgr)
        dali_out = self._run_dali_pipeline(jpeg, 640, 640)
        assert dali_out.shape == (1, 3, 640, 640)

    def test_output_shape_nonsquare(self) -> None:
        bgr = self._make_gradient_image(480, 640)
        _, jpeg = self._make_jpeg_pair(bgr)
        dali_out = self._run_dali_pipeline(jpeg, 608, 960)
        assert dali_out.shape == (1, 3, 608, 960)


@requires_dali
class TestDaliVsUltralyticsReferenceTensors:
    """Compare DALI pipeline against saved ultralytics reference tensors.

    Reference tensors are generated by capture_ultralytics_tensors.py which
    patches ultralytics predictor.preprocess to save the exact float32 tensor
    fed to the model. Tests verify structural match (shape, content position,
    padding) between DALI and the captured ground truth.

    Configs tested:
    - 640_square: imgsz=640, rect=False (standard YOLO inference)
    - 640_rect: imgsz=640, rect=True (variable height, stride-aligned)
    - 608x960_fixed: imgsz=(608,960), rect=False (fixed non-square)
    """

    _TENSORS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "test_fixtures" / "ultralytics_tensors"
    _COCO8_DIR = Path("/home/nkt/datasets/coco8/images/train")

    @staticmethod
    def _run_dali_pipeline(
        jpeg_bytes: bytes,
        target_h: int,
        target_w: int,
    ) -> npt.NDArray[np.floating[Any]]:
        import nvidia.dali.math as dmath  # pyright: ignore[reportMissingImports]
        import nvidia.dali.types as dali_types  # pyright: ignore[reportMissingImports]
        from nvidia.dali import fn  # pyright: ignore[reportMissingImports]
        from nvidia.dali.pipeline import Pipeline  # pyright: ignore[reportMissingImports]

        pipe = Pipeline(batch_size=1, num_threads=4, device_id=0)
        with pipe:
            enc = fn.external_source(device="cpu", name="INPUT_0")
            img = fn.decoders.image(enc, device="cpu", output_type=dali_types.BGR)
            shapes = fn.peek_image_shape(enc)
            oh = fn.slice(shapes, 0, 1, axes=[0])
            ow = fn.slice(shapes, 1, 1, axes=[0])
            s = dmath.min(
                target_h / fn.cast(oh, dtype=dali_types.FLOAT),
                target_w / fn.cast(ow, dtype=dali_types.FLOAT),
            )
            nh = fn.cast(fn.cast(oh, dtype=dali_types.FLOAT) * s, dtype=dali_types.INT32)
            nw = fn.cast(fn.cast(ow, dtype=dali_types.FLOAT) * s, dtype=dali_types.INT32)
            resized = fn.resize(
                img,
                resize_x=fn.cast(nw, dtype=dali_types.FLOAT),
                resize_y=fn.cast(nh, dtype=dali_types.FLOAT),
                interp_type=dali_types.INTERP_LINEAR,
            )
            pr = dmath.max(
                target_h / fn.cast(nh, dtype=dali_types.FLOAT),
                target_w / fn.cast(nw, dtype=dali_types.FLOAT),
            )
            padded = fn.paste(resized, paste_x=0.5, paste_y=0.5, ratio=pr, fill_value=114)
            cropped = fn.crop(
                padded,
                crop_h=target_h,
                crop_w=target_w,
                crop_pos_x=0.5,
                crop_pos_y=0.5,
                out_of_bounds_policy="pad",
                fill_values=114,
            )
            rgb = fn.color_space_conversion(cropped, image_type=dali_types.BGR, output_type=dali_types.RGB)
            norm = fn.crop_mirror_normalize(
                rgb,
                mean=[0, 0, 0],
                std=[255, 255, 255],
                output_layout="CHW",
                dtype=dali_types.FLOAT,
            )
            batched = fn.expand_dims(norm, axes=[0])
            pipe.set_outputs(batched)

        pipe.build()
        pipe.feed_input("INPUT_0", [np.frombuffer(jpeg_bytes, dtype=np.uint8)])
        (out,) = pipe.run()
        result: npt.NDArray[np.floating[Any]] = np.array(out[0])
        return result

    @staticmethod
    def _find_content_bounds(tensor: npt.NDArray[np.floating[Any]]) -> tuple[int, int, int, int]:
        pad_val = 114.0 / 255.0
        if tensor.ndim == 4:
            tensor = tensor[0]
        hwc = np.transpose(tensor, (1, 2, 0))
        non_pad = np.where(~np.all(np.isclose(hwc, pad_val, atol=0.02), axis=2))
        if len(non_pad[0]) == 0:
            return 0, 0, 0, 0
        return int(non_pad[0].min()), int(non_pad[0].max()), int(non_pad[1].min()), int(non_pad[1].max())

    def _load_config(self, label: str) -> dict[str, Any]:
        import json

        meta_path = self._TENSORS_DIR / f"{label}_meta.json"
        if not meta_path.exists():
            pytest.skip(f"Reference tensors not found: {meta_path}. Run capture_ultralytics_tensors.py first")
        return json.loads(meta_path.read_text())  # type: ignore[no-any-return]

    def _run_comparison(self, label: str) -> None:
        if not self._COCO8_DIR.is_dir():
            pytest.skip("coco8 not downloaded")

        meta = self._load_config(label)

        for img_name, info in meta.items():
            img_path = self._COCO8_DIR / img_name
            if not img_path.exists():
                pytest.skip(f"coco8 image not found: {img_path}")

            ref_tensor = np.load(self._TENSORS_DIR / info["npy"])
            _, _, target_h, target_w = ref_tensor.shape

            jpeg_bytes = img_path.read_bytes()
            dali_bchw = self._run_dali_pipeline(jpeg_bytes, target_h, target_w)
            dali_chw = dali_bchw[0] if dali_bchw.ndim == 4 else dali_bchw
            ref_chw = ref_tensor[0]

            assert dali_chw.shape == ref_chw.shape, f"{img_name}: shape {dali_chw.shape} != ref {ref_chw.shape}"

            ref_y0, ref_y1, ref_x0, ref_x1 = self._find_content_bounds(ref_chw)
            dali_y0, dali_y1, dali_x0, dali_x1 = self._find_content_bounds(dali_chw)

            assert abs(dali_y0 - ref_y0) <= 1, f"{img_name}: top {dali_y0} vs ref {ref_y0}"
            assert abs(dali_y1 - ref_y1) <= 1, f"{img_name}: bottom {dali_y1} vs ref {ref_y1}"
            assert abs(dali_x0 - ref_x0) <= 1, f"{img_name}: left {dali_x0} vs ref {ref_x0}"
            assert abs(dali_x1 - ref_x1) <= 1, f"{img_name}: right {dali_x1} vs ref {ref_x1}"

            assert dali_chw.min() >= 0.0, f"{img_name}: min {dali_chw.min()}"
            assert dali_chw.max() <= 1.0, f"{img_name}: max {dali_chw.max()}"

            pad_val = 114.0 / 255.0
            if ref_y0 > 2:
                dali_pad = dali_chw[:, : ref_y0 - 1, :]
                np.testing.assert_allclose(
                    dali_pad,
                    pad_val,
                    atol=0.01,
                    err_msg=f"{img_name}: top padding mismatch",
                )

    def test_640_square(self) -> None:
        self._run_comparison("640_square")

    def test_640_rect(self) -> None:
        self._run_comparison("640_rect")

    def test_608x960_fixed(self) -> None:
        self._run_comparison("608x960_fixed")
