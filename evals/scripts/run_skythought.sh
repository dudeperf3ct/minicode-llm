#!/usr/bin/env bash
# Run LiveCodeBench v5 Easy, Medium, and Hard with SkyThought.
#
# Usage:
#   ./scripts/run_skythought.sh MODEL RESULT_DIR [PROFILE] [BASE_URL]
#
# Examples:
#   ./scripts/run_skythought.sh \
#     Qwen/Qwen3.5-4B-Base \
#     results/qwen3.5-4b-base/livecodebench
#
#   ./scripts/run_skythought.sh \
#     qwen35-4b-post-thinking \
#     results/qwen3.5-4b-post-trained/thinking/livecodebench \
#     thinking \
#     http://127.0.0.1:8001/v1

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 MODEL RESULT_DIR [direct|thinking] [BASE_URL]" >&2
    exit 2
fi

MODEL="$1"
RESULT_DIR="$2"
PROFILE="${3:-direct}"
BASE_URL="${4:-}"

case "$PROFILE" in
    direct)
        SAMPLING_PARAMS="temperature=0,top_p=1,max_tokens=16384"
        ;;
    thinking)
        SAMPLING_PARAMS="temperature=0.6,top_p=0.95,max_tokens=32768"
        [[ -n "$BASE_URL" ]] || {
            echo "error: thinking profile requires BASE_URL" >&2
            exit 2
        }
        ;;
    *)
        echo "error: profile must be direct or thinking" >&2
        exit 2
        ;;
esac

if [[ -n "$BASE_URL" ]]; then
    BACKEND_ARGS=(
        --backend openai
        --backend-args "api_key=${OPENAI_API_KEY:-EMPTY},base_url=$BASE_URL"
    )
else
    BACKEND_ARGS=(
        --backend vllm
        --backend-args tensor_parallel_size=1
    )
fi

command -v uv >/dev/null || {
    echo "error: uv is not installed or not in PATH" >&2
    exit 127
}

mkdir -p "$RESULT_DIR"

for TASK in \
    livecodebench_easy \
    livecodebench_medium \
    livecodebench_hard
do
    echo "==> Running SkyThought: $TASK ($PROFILE)"

    uv run --no-sync --project "$PROJECT_DIR" skythought evaluate \
        --model "$MODEL" \
        --task "$TASK" \
        "${BACKEND_ARGS[@]}" \
        --sampling-params "$SAMPLING_PARAMS" \
        --n 1 \
        --result-dir "$RESULT_DIR" \
        2>&1 | tee "$RESULT_DIR/${TASK}.log"
done
