#!/usr/bin/env bash
# Run HumanEval/HumanEval+ and MBPP/MBPP+ with EvalPlus.
#
# Usage:
#   ./run_evalplus.sh MODEL RESULT_DIR
#
# Example:
#   ./run_evalplus.sh \
#     Qwen/Qwen3.5-4B-Base \
#     results/qwen3.5-4b-base/evalplus

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 MODEL RESULT_DIR" >&2
    exit 2
fi

MODEL="$1"
RESULT_DIR="$2"

if ! command -v evalplus.evaluate >/dev/null 2>&1; then
    echo "error: evalplus.evaluate is not installed or not in PATH" >&2
    exit 127
fi

mkdir -p "$RESULT_DIR"

for DATASET in humaneval mbpp; do
    echo "==> Running EvalPlus: $DATASET"

    evalplus.evaluate \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --backend vllm \
        --greedy \
        --tp 1 \
        --dtype bfloat16 \
        --root "$RESULT_DIR" \
        2>&1 | tee "$RESULT_DIR/${DATASET}.log"
done
