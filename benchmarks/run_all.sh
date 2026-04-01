#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCHMARK_DIR="$REPO_ROOT/benchmarks"
MODEL_REPO="$REPO_ROOT/model_repository"
DATA_DIR="$BENCHMARK_DIR/data"
TRITON_URL="localhost:8001"
NUM_IMAGES=50
WARMUP=5

echo "=== Step 1: Export YOLO models (ONNX) ==="
uv run triton-model-tools export -r "$MODEL_REPO" -m yolo11s yolo26s yolo12s -f onnx

echo ""
echo "=== Step 2: Wait for Triton to load models ==="
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
        echo "Triton is ready"
        break
    fi
    echo "Waiting for Triton... ($i/30)"
    sleep 10
done

echo ""
echo "=== Step 3: Check loaded models ==="
curl -s http://localhost:8000/v2/models | python3 -m json.tool || true

echo ""
echo "=== Step 4: Download COCO val2017 (limited) ==="
uv run triton-model-tools download-coco --target-dir "$DATA_DIR" --max-images "$NUM_IMAGES"

echo ""
echo "=== Step 5: Benchmark yolo11s (ONNX) ==="
uv run triton-model-tools benchmark \
    --image-dir "$DATA_DIR/val2017" \
    -m yolo11s \
    --backend onnx \
    --triton-url "$TRITON_URL" \
    --num-images "$NUM_IMAGES" \
    --warmup-runs "$WARMUP" \
    -o "$BENCHMARK_DIR/yolo11s_onnx.json"

echo ""
echo "=== Step 6: Benchmark yolo26s (ONNX, end2end) ==="
uv run triton-model-tools benchmark \
    --image-dir "$DATA_DIR/val2017" \
    -m yolo26s \
    --backend onnx \
    --triton-url "$TRITON_URL" \
    --num-images "$NUM_IMAGES" \
    --warmup-runs "$WARMUP" \
    --end2end \
    -o "$BENCHMARK_DIR/yolo26s_onnx.json"

echo ""
echo "=== Step 7: Benchmark yolo12s (ONNX) ==="
uv run triton-model-tools benchmark \
    --image-dir "$DATA_DIR/val2017" \
    -m yolo12s \
    --backend onnx \
    --triton-url "$TRITON_URL" \
    --num-images "$NUM_IMAGES" \
    --warmup-runs "$WARMUP" \
    -o "$BENCHMARK_DIR/yolo12s_onnx.json"

echo ""
echo "=== Done! Results saved to $BENCHMARK_DIR ==="
ls -lh "$BENCHMARK_DIR"/*.json
