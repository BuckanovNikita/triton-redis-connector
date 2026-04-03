#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${1:-model_repository}"
TRITON_URL="${2:-localhost:8000}"

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

if [ ! -d "$REPO_PATH" ]; then
    echo "ERROR: Model repository not found: $REPO_PATH"
    exit 1
fi

if ! curl -sf "http://${TRITON_URL}/v2/health/live" >/dev/null 2>&1; then
    echo "ERROR: Triton not reachable at $TRITON_URL"
    echo "Start Triton first or provide correct URL as second argument."
    exit 1
fi

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a RESULTS=()

for model_dir in "$REPO_PATH"/*/; do
    model_name=$(basename "$model_dir")

    if [ ! -f "${model_dir}config.pbtxt" ]; then
        continue
    fi

    if should_skip "$model_name"; then
        RESULTS+=("SKIP  $model_name (auxiliary)")
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    echo "Loading: $model_name ..."

    if ! curl -sf -X POST "http://${TRITON_URL}/v2/repository/models/${model_name}/load" >/dev/null 2>&1; then
        RESULTS+=("FAIL  $model_name (load request failed)")
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    sleep 2

    status=$(curl -sf "http://${TRITON_URL}/v2/models/${model_name}" 2>/dev/null || echo "")
    if [ -z "$status" ]; then
        RESULTS+=("FAIL  $model_name (status check failed)")
        FAIL_COUNT=$((FAIL_COUNT + 1))
    elif echo "$status" | grep -q '"state":"READY"'; then
        RESULTS+=("PASS  $model_name")
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        state=$(echo "$status" | grep -o '"state":"[^"]*"' | head -1 || echo "UNKNOWN")
        RESULTS+=("FAIL  $model_name ($state)")
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    curl -sf -X POST "http://${TRITON_URL}/v2/repository/models/${model_name}/unload" >/dev/null 2>&1 || true
done

echo ""
echo "=== Validation Results ==="
for result in "${RESULTS[@]}"; do
    echo "  $result"
done

echo ""
echo "Total: $((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))  Pass: $PASS_COUNT  Fail: $FAIL_COUNT  Skip: $SKIP_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
