"""CLI entry point for triton-model-tools: subcommand-based model management."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from triton_model_tools.models import YOLO_MODELS

if TYPE_CHECKING:
    from typing import Any, Protocol

    class SubParsers(Protocol):
        """Protocol for argparse subparsers action (avoids private _SubParsersAction)."""

        def add_parser(self, name: str, **kwargs: Any) -> argparse.ArgumentParser:  # noqa: ANN401, D102
            ...


AVAILABLE_MODEL_NAMES = ", ".join(v.name for v in YOLO_MODELS)


def _build_export_parser(
    subparsers: SubParsers,
) -> None:
    export_parser = subparsers.add_parser(
        "export",
        help="Export YOLO models for NVIDIA Triton Inference Server",
    )
    export_parser.add_argument(
        "-r",
        "--repository-root",
        type=Path,
        required=True,
        help="Path to Triton model repository",
    )
    export_parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=None,
        help="Model names to export (default: all). Choices: " + AVAILABLE_MODEL_NAMES,
    )
    export_parser.add_argument(
        "-f",
        "--format",
        choices=["onnx", "tensorrt", "all"],
        default="onnx",
        dest="export_format",
        help="Export format (default: onnx)",
    )
    export_parser.add_argument(
        "--dali-ensemble",
        "--dali",
        action="store_true",
        default=False,
        help="Also generate DALI preprocess + ensemble models",
    )
    export_parser.add_argument(
        "--nms",
        action="store_true",
        default=False,
        help="Also generate NMS postprocess model for non-end2end models",
    )
    export_parser.set_defaults(func=_run_export)


def _run_export(args: argparse.Namespace) -> None:
    from triton_model_tools.export.exporter import export_all_models  # noqa: PLC0415

    repository_root: Path = args.repository_root.resolve()
    model_names: list[str] | None = args.models
    export_format: str = args.export_format
    generate_dali: bool = args.dali_ensemble
    generate_nms: bool = args.nms

    logger.info("Triton Model Export Tool")
    logger.info("Repository root: {}", repository_root)
    logger.info("Export format: {}", export_format)

    if model_names:
        logger.info("Selected models: {}", ", ".join(model_names))
    else:
        logger.info("Exporting all models")

    if generate_dali:
        logger.info("DALI ensemble generation enabled")

    if generate_nms:
        logger.info("NMS postprocess generation enabled")

    exported_dirs = export_all_models(
        repository_root=repository_root,
        model_names=model_names,
        fmt=export_format,  # type: ignore[arg-type]
    )

    nms_model_name: str | None = None
    if generate_nms:
        from triton_model_tools.export.nms import (  # noqa: PLC0415
            NMS_MODEL_NAME,
            NmsModelConfig,
            build_nms_model_directory,
        )

        build_nms_model_directory(
            repository_root=repository_root,
            config=NmsModelConfig(),
        )
        nms_model_name = NMS_MODEL_NAME

    if generate_dali:
        from triton_model_tools.export.dali import (  # noqa: PLC0415
            DaliPipelineConfig,
            build_ensemble_directory,
        )

        end2end_names = {v.name for v in YOLO_MODELS if v.end2end_export}
        dali_config = DaliPipelineConfig()
        for model_dir in exported_dirs:
            effective_nms = None if model_dir.name in end2end_names else nms_model_name
            build_ensemble_directory(
                repository_root=repository_root,
                model_name=model_dir.name,
                config=dali_config,
                nms_model_name=effective_nms,
            )


def _build_infer_parser(
    subparsers: SubParsers,
) -> None:
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run YOLO inference on a folder of images via Triton",
    )
    infer_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to run inference on",
    )
    infer_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Triton model name (e.g. yolo11s, yolo26s)",
    )
    infer_parser.add_argument(
        "--triton-url",
        type=str,
        default="localhost:8001",
        help="Triton gRPC URL (default: localhost:8001)",
    )
    infer_parser.add_argument(
        "--target-size",
        type=int,
        default=640,
        help="Model input size (default: 640)",
    )
    infer_parser.add_argument(
        "--end2end",
        action="store_true",
        default=False,
        help="Model uses end2end NMS output format",
    )
    infer_parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    infer_parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    infer_parser.set_defaults(func=_run_infer)


def _run_infer(args: argparse.Namespace) -> None:
    from triton_model_tools.inference.folder_inference import run_folder_inference  # noqa: PLC0415

    image_dir: Path = args.image_dir.resolve()
    model_name: str = args.model
    triton_url: str = args.triton_url
    target_size: int = args.target_size
    end2end: bool = args.end2end
    conf_threshold: float = args.conf_threshold
    iou_threshold: float = args.iou_threshold

    logger.info("Triton Folder Inference")
    logger.info("Image dir: {}", image_dir)
    logger.info("Model: {}", model_name)
    logger.info("Triton URL: {}", triton_url)

    result = run_folder_inference(
        image_dir=image_dir,
        model_name=model_name,
        triton_url=triton_url,
        target_size=target_size,
        end2end=end2end,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    logger.info("Results: {} images processed", len(result.results))
    logger.info("Total time: {:.2f}s", result.total_time_s)
    logger.info("Average time: {:.1f}ms per image", result.avg_time_ms)

    for img_result in result.results:
        logger.info(
            "  {}: {} detections ({:.1f}ms)",
            img_result.image_path,
            len(img_result.detections),
            img_result.inference_time_ms,
        )
        for det in img_result.detections:
            logger.info(
                "    class={} conf={:.3f} bbox=({:.0f},{:.0f},{:.0f},{:.0f})",
                det.class_id,
                det.confidence,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
            )


def _build_download_coco_parser(
    subparsers: SubParsers,
) -> None:
    download_parser = subparsers.add_parser(
        "download-coco",
        help="Download COCO val2017 images for benchmarking",
    )
    download_parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data"),
        help="Target directory for COCO dataset (default: ./data)",
    )
    download_parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of extracted images (for quick testing)",
    )
    download_parser.set_defaults(func=_run_download_coco)


def _run_download_coco(args: argparse.Namespace) -> None:
    from triton_model_tools.benchmarks.dataset import download_coco_val  # noqa: PLC0415

    target_dir: Path = args.target_dir.resolve()
    max_images: int | None = args.max_images

    logger.info("COCO val2017 Downloader")
    logger.info("Target directory: {}", target_dir)
    if max_images is not None:
        logger.info("Max images: {}", max_images)

    image_dir = download_coco_val(target_dir, max_images=max_images)
    logger.info("Images available at: {}", image_dir)


def _build_benchmark_parser(
    subparsers: SubParsers,
) -> None:
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run inference benchmark on images via Triton",
    )
    bench_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to benchmark with",
    )
    bench_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Triton model name (e.g. yolo11s)",
    )
    bench_parser.add_argument(
        "--backend",
        choices=["onnx", "tensorrt"],
        default="onnx",
        help="Model backend (default: onnx)",
    )
    bench_parser.add_argument(
        "--triton-url",
        type=str,
        default="localhost:8001",
        help="Triton gRPC URL (default: localhost:8001)",
    )
    bench_parser.add_argument(
        "--target-size",
        type=int,
        default=640,
        help="Model input size (default: 640)",
    )
    bench_parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to benchmark (default: 100)",
    )
    bench_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    bench_parser.add_argument(
        "--end2end",
        action="store_true",
        default=False,
        help="Model uses end2end NMS output format",
    )
    bench_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (JSON or CSV based on extension)",
    )
    bench_parser.set_defaults(func=_run_benchmark)


def _run_benchmark(args: argparse.Namespace) -> None:
    from triton_model_tools.benchmarks.benchmark import (  # noqa: PLC0415
        BenchmarkConfig,
        BenchmarkSuite,
        run_benchmark,
        save_benchmark_output,
    )

    image_dir: Path = args.image_dir.resolve()
    model_name: str = args.model
    backend: str = args.backend
    triton_url: str = args.triton_url
    target_size: int = args.target_size
    num_images: int = args.num_images
    warmup_runs: int = args.warmup_runs
    end2end: bool = args.end2end
    output_path: Path | None = args.output

    logger.info("Triton Benchmark")
    logger.info("Image dir: {}", image_dir)
    logger.info("Model: {} (backend: {})", model_name, backend)

    config = BenchmarkConfig(
        model_name=model_name,
        backend=backend,  # type: ignore[arg-type]
        triton_url=triton_url,
        target_size=target_size,
        num_images=num_images,
        warmup_runs=warmup_runs,
        end2end=end2end,
    )

    result = run_benchmark(config, image_dir)
    suite = BenchmarkSuite(results=[result])

    logger.info(
        "Results: {} images, avg={:.2f}ms, throughput={:.1f}fps",
        result.num_images,
        result.avg_latency_ms,
        result.throughput_fps,
    )
    logger.info(
        "Latencies: p50={:.2f}ms p95={:.2f}ms p99={:.2f}ms",
        result.p50_latency_ms,
        result.p95_latency_ms,
        result.p99_latency_ms,
    )

    if output_path is not None:
        save_benchmark_output(suite, output_path.resolve())


def _build_e2e_test_parser(
    subparsers: SubParsers,
) -> None:
    e2e_parser = subparsers.add_parser(
        "e2e-test",
        help="Run E2E test: push images to Redis, connector processes via Triton, collect results",
    )
    e2e_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to push through the connector",
    )
    e2e_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Triton model name (e.g. yolo11s)",
    )
    e2e_parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://redis:6379",
        help="Redis URL (default: redis://redis:6379)",
    )
    e2e_parser.add_argument(
        "--input-stream",
        type=str,
        default="camera_images",
        help="Redis input stream name (default: camera_images)",
    )
    e2e_parser.add_argument(
        "--output-stream",
        type=str,
        default="detection_results",
        help="Redis output stream name (default: detection_results)",
    )
    e2e_parser.add_argument(
        "--consumer-group",
        type=str,
        default="e2e-test",
        help="Redis consumer group name (default: e2e-test)",
    )
    e2e_parser.add_argument(
        "--target-size",
        type=int,
        default=640,
        help="Model input size (default: 640)",
    )
    e2e_parser.add_argument(
        "--end2end",
        action="store_true",
        default=False,
        help="Model uses end2end NMS output format",
    )
    e2e_parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for collecting results (default: 30.0)",
    )
    e2e_parser.set_defaults(func=_run_e2e_test)


def _run_e2e_test(args: argparse.Namespace) -> None:
    from triton_model_tools.testing.e2e_test import (  # noqa: PLC0415
        ConnectorE2ETestHarness,
        E2ETestConfig,
    )

    image_dir: Path = args.image_dir.resolve()
    model_name: str = args.model
    redis_url: str = args.redis_url
    input_stream: str = args.input_stream
    output_stream: str = args.output_stream
    consumer_group: str = args.consumer_group
    target_size: int = args.target_size
    end2end: bool = args.end2end
    timeout_s: float = args.timeout

    logger.info("E2E Connector Test")
    logger.info("Image dir: {}", image_dir)
    logger.info("Model: {}", model_name)
    logger.info("Redis: {}", redis_url)
    logger.info("Streams: {} -> {}", input_stream, output_stream)

    config = E2ETestConfig(
        redis_url=redis_url,
        input_stream=input_stream,
        output_stream=output_stream,
        model_name=model_name,
        consumer_group=consumer_group,
        target_size=target_size,
        end2end=end2end,
        timeout_s=timeout_s,
    )

    harness = ConnectorE2ETestHarness(config)
    result = harness.run_through_connector(image_dir)

    logger.info("Results: {} images processed", len(result.results))
    logger.info("Total time: {:.2f}s", result.total_time_s)
    logger.info("Average time: {:.1f}ms per image", result.avg_time_ms)

    for img_result in result.results:
        logger.info(
            "  {}: {} detections ({:.1f}ms)",
            img_result.image_path,
            len(img_result.detections),
            img_result.inference_time_ms,
        )
        for det in img_result.detections:
            logger.info(
                "    class={} conf={:.3f} bbox=({:.0f},{:.0f},{:.0f},{:.0f})",
                det.class_id,
                det.confidence,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
            )


def _build_pipeline_benchmark_parser(
    subparsers: SubParsers,
) -> None:
    pb_parser = subparsers.add_parser(
        "pipeline-benchmark",
        help="Measure E2E performance of the Redis detection pipeline",
    )
    pb_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to benchmark with",
    )
    pb_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Triton model name (e.g. yolo11s)",
    )
    pb_parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://redis:6379",
        help="Redis URL (default: redis://redis:6379)",
    )
    pb_parser.add_argument(
        "--input-stream",
        type=str,
        default="camera_images",
        help="Redis input stream name (default: camera_images)",
    )
    pb_parser.add_argument(
        "--output-stream",
        type=str,
        default="detection_results",
        help="Redis output stream name (default: detection_results)",
    )
    pb_parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations (default: 3)",
    )
    pb_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup iterations (default: 1)",
    )
    pb_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for collecting results per iteration (default: 60.0)",
    )
    pb_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path for results JSON",
    )
    pb_parser.set_defaults(func=_run_pipeline_benchmark)


def _run_pipeline_benchmark(args: argparse.Namespace) -> None:
    from triton_model_tools.benchmarks.pipeline_benchmark import (  # noqa: PLC0415
        PipelineBenchmarkConfig,
        run_pipeline_benchmark,
    )

    image_dir: Path = args.image_dir.resolve()
    model_name: str = args.model
    redis_url: str = args.redis_url
    input_stream: str = args.input_stream
    output_stream: str = args.output_stream
    num_iterations: int = args.num_iterations
    warmup_runs: int = args.warmup_runs
    timeout_s: float = args.timeout
    output_path: Path | None = args.output

    logger.info("Pipeline E2E Benchmark")
    logger.info("Image dir: {}", image_dir)
    logger.info("Model: {}", model_name)
    logger.info("Redis: {}", redis_url)

    config = PipelineBenchmarkConfig(
        redis_url=redis_url,
        input_stream=input_stream,
        output_stream=output_stream,
        model_name=model_name,
        num_iterations=num_iterations,
        warmup_runs=warmup_runs,
        timeout_s=timeout_s,
    )

    result = run_pipeline_benchmark(config, image_dir)

    logger.info(
        "Results: {} images, avg={:.2f}ms, throughput={:.1f}fps",
        result.num_images,
        result.avg_latency_ms,
        result.throughput_fps,
    )
    logger.info(
        "Latencies: p50={:.2f}ms p95={:.2f}ms p99={:.2f}ms",
        result.p50_latency_ms,
        result.p95_latency_ms,
        result.p99_latency_ms,
    )

    if output_path is not None:
        import json  # noqa: PLC0415

        resolved_output = output_path.resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        with resolved_output.open("w") as f:
            json.dump(result.model_dump(), f, indent=2)
        logger.info("Benchmark results saved to {}", resolved_output)


def _build_compare_folders_parser(
    subparsers: SubParsers,
) -> None:
    cf_parser = subparsers.add_parser(
        "compare-folders",
        help="Compare detection txt results from two folders (baseline vs candidate)",
    )
    cf_parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Directory with baseline detection txt files",
    )
    cf_parser.add_argument(
        "--candidate-dir",
        type=Path,
        required=True,
        help="Directory with candidate detection txt files",
    )
    cf_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model name for detection results",
    )
    cf_parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for detection matching (default: 0.5)",
    )
    cf_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for comparison results JSON",
    )
    cf_parser.set_defaults(func=_run_compare_folders)


def _run_compare_folders(args: argparse.Namespace) -> None:
    import json as json_mod  # noqa: PLC0415

    from triton_model_tools.benchmarks.comparison import compare_inference  # noqa: PLC0415
    from triton_model_tools.inference.detection_io import load_folder_detections  # noqa: PLC0415

    baseline_dir: Path = args.baseline_dir.resolve()
    candidate_dir: Path = args.candidate_dir.resolve()
    model_name: str = args.model
    iou_threshold: float = args.iou_threshold
    output_path: Path | None = args.output

    logger.info("Folder Comparison")
    logger.info("Baseline: {}", baseline_dir)
    logger.info("Candidate: {}", candidate_dir)
    logger.info("Model: {}", model_name)

    baseline = load_folder_detections(baseline_dir, model_name)
    candidate = load_folder_detections(candidate_dir, model_name)
    summary = compare_inference(baseline, candidate, iou_threshold)

    logger.info("Overall precision: {:.4f}", summary.overall.precision)
    logger.info("Overall recall: {:.4f}", summary.overall.recall)
    logger.info("Overall avg IoU: {:.4f}", summary.overall.avg_iou)
    logger.info("Class mismatch rate: {:.4f}", summary.overall.class_mismatch_rate)
    logger.info("False positives: {}", summary.overall.false_positive_count)
    logger.info("Missed detections: {}", summary.overall.missed_count)

    if output_path is not None:
        resolved_output = output_path.resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        with resolved_output.open("w") as f:
            json_mod.dump(summary.model_dump(), f, indent=2)
        logger.info("Comparison results saved to {}", resolved_output)


def _build_pipeline_predict_parser(
    subparsers: SubParsers,
) -> None:
    pp_parser = subparsers.add_parser(
        "pipeline-predict",
        help="Run images through Redis pipeline and save detection txt files",
    )
    pp_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to push through the pipeline",
    )
    pp_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save detection txt files",
    )
    pp_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Triton model name (e.g. yolo11s)",
    )
    pp_parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://redis:6379",
        help="Redis URL (default: redis://redis:6379)",
    )
    pp_parser.add_argument(
        "--input-stream",
        type=str,
        default="camera_images",
        help="Redis input stream name (default: camera_images)",
    )
    pp_parser.add_argument(
        "--output-stream",
        type=str,
        default="detection_results",
        help="Redis output stream name (default: detection_results)",
    )
    pp_parser.add_argument(
        "--consumer-group",
        type=str,
        default="pipeline-predict",
        help="Redis consumer group name (default: pipeline-predict)",
    )
    pp_parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for collecting results (default: 30.0)",
    )
    pp_parser.set_defaults(func=_run_pipeline_predict)


def _run_pipeline_predict(args: argparse.Namespace) -> None:
    from triton_model_tools.testing.e2e_test import E2ETestConfig  # noqa: PLC0415
    from triton_model_tools.testing.pipeline_predict import run_pipeline_predict  # noqa: PLC0415

    image_dir: Path = args.image_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    model_name: str = args.model
    redis_url: str = args.redis_url
    input_stream: str = args.input_stream
    output_stream: str = args.output_stream
    consumer_group: str = args.consumer_group
    timeout_s: float = args.timeout

    logger.info("Pipeline Prediction")
    logger.info("Image dir: {}", image_dir)
    logger.info("Output dir: {}", output_dir)
    logger.info("Model: {}", model_name)
    logger.info("Redis: {}", redis_url)

    config = E2ETestConfig(
        redis_url=redis_url,
        input_stream=input_stream,
        output_stream=output_stream,
        model_name=model_name,
        consumer_group=consumer_group,
        timeout_s=timeout_s,
    )

    result = run_pipeline_predict(image_dir, output_dir, config)

    logger.info("Results: {} images processed", len(result.results))
    logger.info("Total time: {:.2f}s", result.total_time_s)
    logger.info("Average time: {:.1f}ms per image", result.avg_time_ms)


def _build_predict_parser(
    subparsers: SubParsers,
) -> None:
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run ultralytics YOLO prediction on an image folder and save txt results",
    )
    predict_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to predict on",
    )
    predict_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save detection txt files",
    )
    predict_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Ultralytics model path or name (default: yolo11s.pt)",
    )
    predict_parser.add_argument(
        "--target-size",
        type=int,
        default=640,
        help="Model input size (default: 640)",
    )
    predict_parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    predict_parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    predict_parser.set_defaults(func=_run_predict)


def _run_predict(args: argparse.Namespace) -> None:
    from triton_model_tools.inference.predict import PredictConfig, run_ultralytics_predict  # noqa: PLC0415

    image_dir: Path = args.image_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    model: str = args.model
    target_size: int = args.target_size
    conf_threshold: float = args.conf_threshold
    iou_threshold: float = args.iou_threshold

    logger.info("Ultralytics YOLO Prediction")
    logger.info("Image dir: {}", image_dir)
    logger.info("Output dir: {}", output_dir)
    logger.info("Model: {}", model)

    config = PredictConfig(
        model_path=model,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        target_size=target_size,
    )

    result = run_ultralytics_predict(image_dir, output_dir, config)

    logger.info("Results: {} images processed", len(result.results))
    logger.info("Total time: {:.2f}s", result.total_time_s)
    logger.info("Average time: {:.1f}ms per image", result.avg_time_ms)

    for img_result in result.results:
        logger.info(
            "  {}: {} detections ({:.1f}ms)",
            img_result.image_path,
            len(img_result.detections),
            img_result.inference_time_ms,
        )


def _build_compare_parser(
    subparsers: SubParsers,
) -> None:
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare detection results between baseline and candidate",
    )
    compare_parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline detection results JSON file",
    )
    compare_parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to candidate detection results JSON file",
    )
    compare_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for comparison results JSON",
    )
    compare_parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for detection matching (default: 0.5)",
    )
    compare_parser.set_defaults(func=_run_compare)


def _run_compare(args: argparse.Namespace) -> None:
    import json  # noqa: PLC0415

    from triton_model_tools.benchmarks.comparison import compare_inference  # noqa: PLC0415
    from triton_model_tools.models.detection import ImageDetectionResult  # noqa: PLC0415

    baseline_path: Path = args.baseline.resolve()
    candidate_path: Path = args.candidate.resolve()
    output_path: Path | None = args.output
    iou_threshold: float = args.iou_threshold

    logger.info("Detection Comparison")
    logger.info("Baseline: {}", baseline_path)
    logger.info("Candidate: {}", candidate_path)

    with baseline_path.open() as f:
        baseline_data = json.load(f)
    with candidate_path.open() as f:
        candidate_data = json.load(f)

    baseline = [ImageDetectionResult.model_validate(item) for item in baseline_data]
    candidate = [ImageDetectionResult.model_validate(item) for item in candidate_data]

    summary = compare_inference(baseline, candidate, iou_threshold)

    logger.info("Overall precision: {:.4f}", summary.overall.precision)
    logger.info("Overall recall: {:.4f}", summary.overall.recall)
    logger.info("Overall avg IoU: {:.4f}", summary.overall.avg_iou)
    logger.info("Class mismatch rate: {:.4f}", summary.overall.class_mismatch_rate)
    logger.info("False positives: {}", summary.overall.false_positive_count)
    logger.info("Missed detections: {}", summary.overall.missed_count)

    if output_path is not None:
        resolved_output = output_path.resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        with resolved_output.open("w") as f:
            json.dump(summary.model_dump(), f, indent=2)
        logger.info("Comparison results saved to {}", resolved_output)


def _build_full_benchmark_parser(
    subparsers: SubParsers,
) -> None:
    fb_parser = subparsers.add_parser(
        "full-benchmark",
        help="Run full performance matrix across models and pipeline configs",
    )
    fb_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to benchmark with",
    )
    fb_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to benchmark (e.g. yolo11s yolo26s yolo12s)",
    )
    fb_parser.add_argument(
        "--triton-url",
        type=str,
        default="localhost:8001",
        help="Triton gRPC URL (default: localhost:8001)",
    )
    fb_parser.add_argument(
        "--target-size",
        type=int,
        default=640,
        help="Model input size (default: 640)",
    )
    fb_parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images per benchmark run (default: 100)",
    )
    fb_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup iterations per run (default: 5)",
    )
    fb_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output file path for results (JSON or CSV)",
    )
    fb_parser.set_defaults(func=_run_full_benchmark)


def _run_full_benchmark(args: argparse.Namespace) -> None:
    from triton_model_tools.benchmarks.benchmark import (  # noqa: PLC0415
        build_full_matrix,
        run_benchmark_matrix,
        save_benchmark_output,
    )

    image_dir: Path = args.image_dir.resolve()
    models: list[str] = args.models
    triton_url: str = args.triton_url
    target_size: int = args.target_size
    num_images: int = args.num_images
    warmup_runs: int = args.warmup_runs
    output_path: Path = args.output.resolve()

    logger.info("Full Performance Matrix Benchmark")
    logger.info("Image dir: {}", image_dir)
    logger.info("Models: {}", ", ".join(models))
    logger.info("Output: {}", output_path)

    configs = build_full_matrix(
        models=models,
        triton_url=triton_url,
        target_size=target_size,
        num_images=num_images,
        warmup_runs=warmup_runs,
    )

    suite = run_benchmark_matrix(configs, image_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_benchmark_output(suite, output_path)

    logger.info("Full benchmark complete: {} configurations", len(suite.results))
    for result in suite.results:
        logger.info(
            "  {} ({}+{}): avg={:.2f}ms throughput={:.1f}fps",
            result.config.model_name,
            result.config.backend,
            result.config.preprocess_mode,
            result.avg_latency_ms,
            result.throughput_fps,
        )


def main() -> None:
    """Entry point for triton-model-tools CLI."""
    parser = argparse.ArgumentParser(
        prog="triton-model-tools",
        description="Tools for managing YOLO models on NVIDIA Triton Inference Server",
    )
    subparsers = parser.add_subparsers(dest="command")

    _build_export_parser(subparsers)
    _build_infer_parser(subparsers)
    _build_download_coco_parser(subparsers)
    _build_benchmark_parser(subparsers)
    _build_full_benchmark_parser(subparsers)
    _build_e2e_test_parser(subparsers)
    _build_compare_parser(subparsers)
    _build_predict_parser(subparsers)
    _build_pipeline_predict_parser(subparsers)
    _build_compare_folders_parser(subparsers)
    _build_pipeline_benchmark_parser(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        sys.exit(130)
    except ValueError as exc:
        logger.error("{}", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
