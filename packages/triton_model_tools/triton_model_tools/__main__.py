"""CLI entry point for triton-model-tools: subcommand-based model management."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from triton_model_tools.models import YOLO_MODELS

AVAILABLE_MODEL_NAMES = ", ".join(v.name for v in YOLO_MODELS)


def _build_export_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
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
    from triton_model_tools.exporter import export_all_models  # noqa: PLC0415

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
        from triton_model_tools.nms import (  # noqa: PLC0415
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
        from triton_model_tools.dali import (  # noqa: PLC0415
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
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
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
    from triton_model_tools.folder_inference import run_folder_inference  # noqa: PLC0415

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
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
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
    from triton_model_tools.dataset import download_coco_val  # noqa: PLC0415

    target_dir: Path = args.target_dir.resolve()
    max_images: int | None = args.max_images

    logger.info("COCO val2017 Downloader")
    logger.info("Target directory: {}", target_dir)
    if max_images is not None:
        logger.info("Max images: {}", max_images)

    image_dir = download_coco_val(target_dir, max_images=max_images)
    logger.info("Images available at: {}", image_dir)


def _build_benchmark_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
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
    from triton_model_tools.benchmark import (  # noqa: PLC0415
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
