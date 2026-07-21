#!/usr/bin/env bash
# Run LiveCodeBench v5 Easy, Medium, and Hard with SkyThought on one H100.
#
# Before running, ensure the SkyThought LiveCodeBench task YAML files use:
#   version_tag: release_v5
#
# Usage:
#   ./run_skythought.sh MODEL RESULT_DIR
#
# Example:
#   ./run_skythought.sh \
#     Qwen/Qwen3.5-4B-Base \
#     results/qwen3.5-4b-base/livecodebench

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 MODEL RESULT_DIR" >&2
    exit 2
fi

MODEL="$1"
RESULT_DIR="$2"

if ! command -v skythought >/dev/null 2>&1; then
    echo "error: skythought is not installed or not in PATH" >&2
    exit 127
fi

mkdir -p "$RESULT_DIR"

for TASK in \
    livecodebench_easy \
    livecodebench_medium \
    livecodebench_hard
do
    echo "==> Running SkyThought: $TASK"

    skythought evaluate \
        --model "$MODEL" \
        --task "$TASK" \
        --backend vllm \
        --backend-args tensor_parallel_size=1 \
        --sampling-params temperature=0,top_p=1,max_tokens=16384 \
        --n 1 \
        --result-dir "$RESULT_DIR" \
        2>&1 | tee "$RESULT_DIR/${TASK}.log"
done
