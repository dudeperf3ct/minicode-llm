#!/usr/bin/env bash
# Run HumanEval/HumanEval+ and MBPP/MBPP+ with EvalPlus.
#
# Usage:
#   ./run_evalplus.sh MODEL RESULT_DIR [OPTIONS]
#
# Options:
#   --profile direct|thinking
#   --base-url URL
#   --max-new-tokens N
#
# Existing two-argument calls use the original direct evaluation protocol.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: ./run_evalplus.sh MODEL RESULT_DIR [OPTIONS]

Options:
  --profile direct|thinking  Evaluation profile (default: direct)
  --base-url URL             OpenAI-compatible vLLM endpoint
  --max-new-tokens N         Override the profile generation limit
  -h, --help                 Show this help

The direct profile defaults to greedy decoding and 768 generated tokens.
Without --base-url it uses EvalPlus's embedded vLLM backend, preserving the
original evaluation protocol. The thinking profile requires --base-url and
defaults to temperature 0.6 and 32,768 generated tokens.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

MODEL="$1"
RESULT_DIR="$2"
shift 2

PROFILE="direct"
BASE_URL=""
MAX_NEW_TOKENS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            [[ $# -ge 2 ]] || { echo "error: --profile requires a value" >&2; exit 2; }
            PROFILE="$2"
            shift 2
            ;;
        --base-url)
            [[ $# -ge 2 ]] || { echo "error: --base-url requires a value" >&2; exit 2; }
            BASE_URL="$2"
            shift 2
            ;;
        --max-new-tokens)
            [[ $# -ge 2 ]] || { echo "error: --max-new-tokens requires a value" >&2; exit 2; }
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

case "$PROFILE" in
    direct)
        TEMPERATURE="0"
        DEFAULT_MAX_NEW_TOKENS="768"
        GREEDY_ARGS=(--greedy)
        ;;
    thinking)
        TEMPERATURE="0.6"
        DEFAULT_MAX_NEW_TOKENS="32768"
        GREEDY_ARGS=()
        if [[ -z "$BASE_URL" ]]; then
            echo "error: the thinking profile requires --base-url" >&2
            exit 2
        fi
        ;;
    *)
        echo "error: unsupported profile: $PROFILE" >&2
        exit 2
        ;;
esac

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-$DEFAULT_MAX_NEW_TOKENS}"

if [[ ! "$MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --max-new-tokens must be a positive integer" >&2
    exit 2
fi

if [[ -n "$BASE_URL" ]]; then
    BACKEND="openai"
    BACKEND_ARGS=(--base-url "$BASE_URL")
else
    BACKEND="vllm"
    BACKEND_ARGS=()
fi

if ! command -v python >/dev/null 2>&1; then
    echo "error: python is not installed or not in PATH" >&2
    exit 127
fi

if ! python -c "import evalplus" >/dev/null 2>&1; then
    echo "error: evalplus is not installed in the active Python environment" >&2
    exit 127
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RESULT_DIR"

for DATASET in humaneval mbpp; do
    echo "==> Running EvalPlus: $DATASET ($PROFILE profile)"

    python "$SCRIPT_DIR/run_evalplus.py" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --result-dir "$RESULT_DIR" \
        --profile "$PROFILE" \
        --backend "$BACKEND" \
        --temperature "$TEMPERATURE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        "${BACKEND_ARGS[@]}" \
        "${GREEDY_ARGS[@]}" \
        2>&1 | tee "$RESULT_DIR/${DATASET}.log"
done
