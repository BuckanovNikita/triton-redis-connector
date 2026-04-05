"""Benchmark pool variant preprocessing throughput.

Measures end-to-end throughput with different num_workers settings
for PreprocessorNode (BaseProcessPoolProcessor).

Usage:
    uv run python scripts/bench_pool.py --image-dir benchmarks/data/val2017
"""

import argparse
import sys
import threading
import time
from pathlib import Path

from loguru import logger
from pydantic import ConfigDict
from typed_flow.config.schema import AppConfig, QueueConfig
from typed_flow.engine.app import Application
from typed_flow.errors.models import PipelineError
from typed_flow.protocols import (
    BaseProcessPoolProcessor,
    BaseThreadSink,
    BaseThreadSource,
)

from triton_redis_connector.config.schema import DecoderType
from triton_redis_connector.engine.events import ImagePayload, TritonRequest
from triton_redis_connector.errors.models import InferenceError, PreprocessingError
from triton_redis_connector.inference.request import InferenceOutput
from triton_redis_connector.preprocessing.decoder import create_decoder


class BenchImageEvent(ImagePayload):
    model_config = ConfigDict(frozen=True)


class BenchPreprocessor(BaseProcessPoolProcessor[BenchImageEvent, TritonRequest | PreprocessingError]):
    def __init__(
        self,
        decoder_type: DecoderType,
        target_size: int,
        model_name: str,
        num_workers: int = 8,
    ) -> None:
        super().__init__(num_workers=num_workers)
        self._decoder_type = decoder_type
        self._target_size = target_size
        self._model_name = model_name
        self._decoder: object | None = None
        self._preprocessor: object | None = None

    def _ensure_initialized(self) -> tuple[object, object]:
        if self._decoder is None:
            from triton_shared_models.detection.preprocessing import LetterBoxPreprocessor  # noqa: PLC0415

            self._decoder = create_decoder(self._decoder_type)
            self._preprocessor = LetterBoxPreprocessor(self._target_size)
        return self._decoder, self._preprocessor

    def __call__(self, event: BenchImageEvent) -> TritonRequest | PreprocessingError | None:
        from typed_flow.protocols.metrics import TimeMetric

        decoder, preprocessor = self._ensure_initialized()

        if event.image_bytes is None:
            return None

        try:
            decoded = decoder.decode(event.image_bytes)  # type: ignore[attr-defined]
        except ValueError as exc:
            return PreprocessingError(
                node_name="bench_preprocess",
                message_id=event.flow_meta.message_id,
                decoder_type="decode",
                error_type="DecodeError",
                error_message=str(exc),
                timestamp_ms=TimeMetric.now(),
            )

        tensor, _scale, _pad_offset = preprocessor.preprocess(decoded)  # type: ignore[attr-defined]
        return TritonRequest(
            flow_meta=event.flow_meta,
            model_name=self._model_name,
            model_version="",
            raw_image=tensor,
        )


class BenchInference(BaseProcessPoolProcessor[TritonRequest, InferenceOutput | InferenceError]):
    """Process-pool inference node — each child process creates its own gRPC client."""

    def __init__(self, grpc_url: str, num_workers: int = 2) -> None:
        super().__init__(num_workers=num_workers)
        self._grpc_url = grpc_url
        self._grpcclient: object | None = None
        self._client: object | None = None
        self._metadata_cache: object | None = None

    def _ensure_client(self) -> tuple[object, object, object]:
        if self._grpcclient is None:
            import tritonclient.grpc as grpcclient  # noqa: PLC0415

            from triton_redis_connector.inference.model_metadata import ModelMetadataCache  # noqa: PLC0415

            self._grpcclient = grpcclient
            self._client = grpcclient.InferenceServerClient(url=self._grpc_url)
            self._metadata_cache = ModelMetadataCache(self._client)
        return self._grpcclient, self._client, self._metadata_cache

    def __call__(self, event: TritonRequest) -> InferenceOutput | InferenceError | None:
        import numpy as np  # noqa: PLC0415
        from typed_flow.protocols.metrics import TimeMetric  # noqa: PLC0415

        start = time.monotonic()
        if event.raw_image is None:
            return None

        grpcclient, client, metadata_cache = self._ensure_client()
        meta = metadata_cache.get(event.model_name, event.model_version)  # type: ignore[attr-defined]
        input_tensor = grpcclient.InferInput(meta.input_name, event.raw_image.shape, "FP32")  # type: ignore[attr-defined]
        input_tensor.set_data_from_numpy(event.raw_image)
        outputs = [grpcclient.InferRequestedOutput(name) for name in meta.output_names]  # type: ignore[attr-defined]

        try:
            result = client.infer(  # type: ignore[attr-defined]
                model_name=event.model_name,
                inputs=[input_tensor],
                outputs=outputs,
                model_version=event.model_version or "",
            )
        except Exception as exc:  # noqa: BLE001
            return InferenceError(
                node_name=f"bench_inference:{event.model_name}",
                model_name=event.model_name,
                model_version=event.model_version,
                error_type=type(exc).__name__,
                error_message=str(exc),
                retry_count=0,
                timestamp_ms=TimeMetric.now(),
            )

        elapsed = time.monotonic() - start
        output_arrays = {name: np.asarray(result.as_numpy(name), dtype=np.float32) for name in meta.output_names}
        return InferenceOutput(
            flow_meta=event.flow_meta,
            model_name=event.model_name,
            model_version=event.model_version,
            outputs=output_arrays,
            inference_time_ms=TimeMetric.from_seconds(elapsed),
        )


class BenchSource(BaseThreadSource[BenchImageEvent]):
    def __init__(self, image_paths: list[Path]) -> None:
        super().__init__()
        self._images = [p.read_bytes() for p in image_paths]
        self._idx = 0
        self._total = 0
        self._limit = 0
        self._lock = threading.Lock()

    def set_limit(self, n: int) -> None:
        with self._lock:
            self._total = 0
            self._limit = n

    def __call__(self) -> BenchImageEvent | None:
        with self._lock:
            if self._total >= self._limit:
                return None
            self._total += 1
        img_bytes = self._images[self._idx % len(self._images)]
        self._idx += 1
        return BenchImageEvent(image_bytes=img_bytes)


class CountingSink(BaseThreadSink[InferenceOutput]):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self._lock = threading.Lock()

    def __call__(self, event: InferenceOutput) -> None:
        with self._lock:
            self.count += 1


class ErrorSink(BaseThreadSink[PipelineError]):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def __call__(self, event: PipelineError) -> None:
        self.count += 1
        logger.warning("Error: {}", type(event).__name__)


def run_benchmark(
    image_dir: Path,
    num_workers: int,
    num_images: int,
    grpc_url: str,
    inference_workers: int = 1,
) -> tuple[float, float, int]:
    image_paths = sorted(image_dir.glob("*.jpg"))[:20]
    if not image_paths:
        logger.error("No .jpg images found in {}", image_dir)
        sys.exit(1)

    source = BenchSource(image_paths)
    preprocessor = BenchPreprocessor(
        decoder_type=DecoderType.OPENCV,
        target_size=640,
        model_name="yolo11s",
        num_workers=num_workers,
    )
    inference = BenchInference(grpc_url=grpc_url, num_workers=inference_workers)
    sink = CountingSink()
    error_sink = ErrorSink()

    config = AppConfig(queues=QueueConfig(maxsize=500))
    app = Application.build(
        config=config,
        processors=[source, preprocessor, inference, sink, error_sink],
    )

    source.set_limit(num_images)

    run_thread = threading.Thread(target=app.run, daemon=True)
    t0 = time.monotonic()
    run_thread.start()

    while sink.count < num_images:
        elapsed = time.monotonic() - t0
        if elapsed > 120:
            logger.error("Timeout after 120s, got {}/{} results", sink.count, num_images)
            break
        time.sleep(0.05)

    elapsed_s = time.monotonic() - t0
    app.shutdown()
    run_thread.join(timeout=10.0)

    throughput = sink.count / elapsed_s if elapsed_s > 0 else 0
    avg_ms = (elapsed_s / sink.count * 1000) if sink.count > 0 else 0
    return throughput, avg_ms, error_sink.count


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pool preprocessing")
    parser.add_argument("--image-dir", type=Path, default=Path("benchmarks/data/val2017"))
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--grpc-url", default="localhost:8001")
    parser.add_argument("--workers", type=str, default="1,2,4,8")
    parser.add_argument("--inference-workers", type=str, default="1,2,4")
    args = parser.parse_args()

    preprocess_counts = [int(w) for w in args.workers.split(",")]
    inference_counts = [int(w) for w in args.inference_workers.split(",")]

    logger.info("Pool preprocessing benchmark")
    logger.info("Image dir: {}", args.image_dir)
    logger.info("Num images: {}", args.num_images)
    logger.info("Preprocess workers: {}", preprocess_counts)
    logger.info("Inference workers: {}", inference_counts)

    results: list[tuple[int, int, float, float, int]] = []

    for iw in inference_counts:
        for pw in preprocess_counts:
            logger.info("--- preprocess={} inference={} ---", pw, iw)
            fps, avg_ms, errors = run_benchmark(
                args.image_dir, pw, args.num_images, args.grpc_url, inference_workers=iw
            )
            results.append((pw, iw, fps, avg_ms, errors))
            logger.info("preprocess={} inference={}: {:.1f} fps, {:.1f} ms/img, {} errors", pw, iw, fps, avg_ms, errors)

    logger.info("")
    logger.info("=== RESULTS ===")
    logger.info("{:<12} {:<12} {:<12} {:<12} {:<12}", "preproc", "inference", "fps", "avg_ms", "errors")
    for pw, iw, fps, avg_ms, errors in results:
        logger.info("{:<12} {:<12} {:<12.1f} {:<12.1f} {:<12}", pw, iw, fps, avg_ms, errors)


if __name__ == "__main__":
    main()
