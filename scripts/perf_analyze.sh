#!/usr/bin/env bash
set -euo pipefail

MODEL=""
PROTOCOL="grpc"
URL="localhost:8001"
CONCURRENCY="1"
OUTPUT_DIR="e2e_results/perf_analyzer"
MEASUREMENT_INTERVAL=10000
SHAPE=""

usage() {
    cat <<USAGE
Usage: $0 -m MODEL [-p PROTOCOL] [-u URL] [-c CONCURRENCY] [-o OUTPUT_DIR]

Run perf_analyzer against a Triton model at various concurrency levels.
Designed to run inside the triton-sdk container.

Options:
  -m MODEL        Model name (required)
  -p PROTOCOL     Protocol: grpc or http (default: grpc)
  -u URL          Triton URL (default: localhost:8001)
  -c CONCURRENCY  Comma-separated concurrency levels (default: 1)
  -o OUTPUT_DIR   Output directory for CSV results (default: e2e_results/perf_analyzer)
  -i INTERVAL     Measurement interval in ms (default: 10000)
  -s SHAPE        Input shape for dynamic dims (e.g. images:3,640,640)
  -h              Show this help

Examples:
  $0 -m yolo11s -u triton:8001 -c 1,4,8 -s images:3,640,640
  $0 -m yolo11s -p http -u triton:8000 -c 1 -o results/
USAGE
    exit 0
}

while getopts "m:p:u:c:o:i:s:h" opt; do
    case $opt in
        m) MODEL="$OPTARG" ;;
        p) PROTOCOL="$OPTARG" ;;
        u) URL="$OPTARG" ;;
        c) CONCURRENCY="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        s) SHAPE="$OPTARG" ;;
        i) MEASUREMENT_INTERVAL="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: Model name is required (-m)"
    usage
fi

if [ "$PROTOCOL" != "grpc" ] && [ "$PROTOCOL" != "http" ]; then
    echo "ERROR: Protocol must be grpc or http, got: $PROTOCOL"
    exit 1
fi

if ! command -v perf_analyzer &>/dev/null; then
    echo "ERROR: perf_analyzer not found. Run this script inside the triton-sdk container:"
    echo "  docker compose -f .devcontainer/docker-compose.debug.yml exec triton-sdk bash"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

IFS=',' read -ra LEVELS <<< "$CONCURRENCY"

HEADER_PRINTED=0

for level in "${LEVELS[@]}"; do
    csv_file="${OUTPUT_DIR}/${MODEL}_c${level}.csv"
    echo "--- Running: model=$MODEL concurrency=$level protocol=$PROTOCOL ---"

    SHAPE_ARG=""
    if [ -n "$SHAPE" ]; then
        SHAPE_ARG="--shape $SHAPE"
    fi

    if ! perf_analyzer \
        -m "$MODEL" \
        -u "$URL" \
        -i "$PROTOCOL" \
        --concurrency-range "$level" \
        --measurement-interval "$MEASUREMENT_INTERVAL" \
        -f "$csv_file" \
        --verbose-csv $SHAPE_ARG 2>&1; then
        echo "FAILED: model=$MODEL concurrency=$level"
        echo "$MODEL,$level,FAILED,-,-,-,-" >> "${OUTPUT_DIR}/${MODEL}_summary.csv"
        continue
    fi

    echo "Saved: $csv_file"

    if [ -f "$csv_file" ] && [ -s "$csv_file" ]; then
        if [ $HEADER_PRINTED -eq 0 ]; then
            echo ""
            echo "=== Summary: $MODEL ==="
            printf "%-12s %-14s %-12s %-12s %-12s %-12s\n" \
                "Concurrency" "Throughput" "Avg(ms)" "P50(ms)" "P95(ms)" "P99(ms)"
            printf "%-12s %-14s %-12s %-12s %-12s %-12s\n" \
                "-----------" "----------" "-------" "-------" "-------" "-------"
            HEADER_PRINTED=1
        fi

        tail -n +2 "$csv_file" | head -1 | while IFS=',' read -r _ _ throughput _ avg_lat _ p50 _ p95 _ p99 _; do
            printf "%-12s %-14s %-12s %-12s %-12s %-12s\n" \
                "$level" "$throughput" "$avg_lat" "$p50" "$p95" "$p99"
        done
    fi
done

echo ""
echo "Results saved to: $OUTPUT_DIR"
