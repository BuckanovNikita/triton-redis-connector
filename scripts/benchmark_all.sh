#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="model_repository"
URL="localhost:8001"
OUTPUT_DIR="e2e_results/benchmark"
IMAGE_DIR=""

SKIP_PATTERNS=(
    "_dali_preprocess"
    "_dali_ensemble"
    "_dali_rescale_ensemble"
    "_dali_nms_rescale_ensemble"
    "rescale_postprocess"
    "nms_postprocess"
)

should_skip() {
    local name="$1"
    for pattern in "${SKIP_PATTERNS[@]}"; do
        if [[ "$name" == *"$pattern"* ]]; then
            return 0
        fi
    done
    return 1
}

usage() {
    cat <<USAGE
Usage: $0 --image-dir DIR [--repository-root DIR] [--url URL] [--output-dir DIR]

Benchmark all models in a Triton model repository.

Options:
  --image-dir DIR          Directory with benchmark images (required)
  --repository-root DIR    Model repository path (default: model_repository)
  --url URL                Triton gRPC URL (default: localhost:8001)
  --output-dir DIR         Output directory (default: e2e_results/benchmark)
  -h, --help               Show this help

Skips auxiliary models (DALI preprocess, ensembles, postprocess).
Includes TRT-accelerated variants (*_ort_trt).

Example:
  $0 --image-dir benchmarks/data/coco_val2017 --repository-root model_repository
USAGE
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --image-dir) IMAGE_DIR="$2"; shift 2 ;;
        --repository-root) REPO_ROOT="$2"; shift 2 ;;
        --url) URL="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$IMAGE_DIR" ]; then
    echo "ERROR: --image-dir is required"
    usage
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "ERROR: Image directory not found: $IMAGE_DIR"
    exit 1
fi

if [ ! -d "$REPO_ROOT" ]; then
    echo "ERROR: Model repository not found: $REPO_ROOT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

declare -a MODELS=()
for model_dir in "$REPO_ROOT"/*/; do
    model_name=$(basename "$model_dir")
    if [ ! -f "${model_dir}config.pbtxt" ]; then
        continue
    fi
    if should_skip "$model_name"; then
        echo "Skipping auxiliary: $model_name"
        continue
    fi
    MODELS+=("$model_name")
done

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models found in $REPO_ROOT"
    exit 1
fi

echo "Found ${#MODELS[@]} model(s) to benchmark: ${MODELS[*]}"
echo ""

PASS_COUNT=0
FAIL_COUNT=0
declare -a RESULTS=()

for model_name in "${MODELS[@]}"; do
    output_file="${OUTPUT_DIR}/${model_name}.json"
    echo "=== Benchmarking: $model_name ==="

    if uv run triton-model-tools benchmark \
        --image-dir "$IMAGE_DIR" \
        -m "$model_name" \
        --triton-url "$URL" \
        --repository-root "$REPO_ROOT" \
        -o "$output_file" 2>&1; then
        RESULTS+=("PASS  $model_name -> $output_file")
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        RESULTS+=("FAIL  $model_name")
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

echo "=== Benchmark Summary ==="
for result in "${RESULTS[@]}"; do
    echo "  $result"
done

echo ""
echo "Total: ${#MODELS[@]}  Pass: $PASS_COUNT  Fail: $FAIL_COUNT"
echo "Results in: $OUTPUT_DIR"
