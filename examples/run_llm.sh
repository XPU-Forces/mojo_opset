#!/bin/bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DEFAULT_LOCAL_PATH="/data08/tbw/mojo_opset_info/Deepseek_v4_int8_w8a8"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}"
    exit 1
fi

MODEL_PATH="${1:-$DEFAULT_LOCAL_PATH}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at ${MODEL_PATH}"
    exit 1
fi

echo "Running inference with model at: ${MODEL_PATH}"

if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
    echo "Current python cannot import torch: ${PYTHON_BIN}"
    exit 1
fi

export MOJO_BACKEND="${MOJO_BACKEND:-torch_npu}"
export MOJO_DISABLE_ASSERTION_REWRITE="${MOJO_DISABLE_ASSERTION_REWRITE:-1}"

cd "$PROJECT_ROOT" || exit 1
"${PYTHON_BIN}" -m examples.llm_inference \
    --model_path "${MODEL_PATH}" \
    --device "${DEVICE:-npu}" \
    --num_layers "${LLM_NUM_LAYERS:-2}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
    --prompt "${PROMPT:-你好}"
