#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DEFAULT_LOCAL_PATH="/data00/dpskv4-flash-quant"

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

EP_SIZE="${EP_SIZE:-8}"
NUM_LAYERS="${LLM_NUM_LAYERS:-43}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
PA_MAX_LENGTH="${PA_MAX_LENGTH:-2048}"
PROMPT="${PROMPT:-你好}"

cd "$PROJECT_ROOT" || exit 1

if [ "$EP_SIZE" -eq 1 ]; then
    echo "EP=1, single card inference"
    ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}" \
    "${PYTHON_BIN}" -m examples.llm_inference \
        --model_path "${MODEL_PATH}" \
        --device "${DEVICE:-npu}" \
        --num_layers "${NUM_LAYERS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --pa_max_length "${PA_MAX_LENGTH}" \
        --prompt "${PROMPT}" \
        --ep_size 1
else
    echo "EP=${EP_SIZE}, multi-card inference"

    MA_NUM_GPUS="$EP_SIZE"

    export WORLD_SIZE="${EP_SIZE}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-6038}"
    export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
    export HCCL_IF_IP="${HCCL_IF_IP:-$(hostname -I | awk '{print $1}')}"
    export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-23456}"
    export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
    export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1200}"
    export RANK_OFFSET="${RANK_OFFSET:-0}"

    echo "HCCL config: MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"
    echo "HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}, HCCL_IF_IP=${HCCL_IF_IP}"
    echo "HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT}, WORLD_SIZE=${WORLD_SIZE}"

    PIDS=()
    for((i=0; i<${MA_NUM_GPUS}; i++))
    do
        export LOCAL_RANK=$i
        export RANK_ID=$(expr $i + $RANK_OFFSET)
        export NPU_DEVICE_IDX=$i

        echo "Launching rank ${RANK_ID} (local_rank=${LOCAL_RANK}, npu_device=${i})..."

        "${PYTHON_BIN}" -m examples.llm_inference \
            --model_path "${MODEL_PATH}" \
            --device "${DEVICE:-npu}" \
            --num_layers "${NUM_LAYERS}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --pa_max_length "${PA_MAX_LENGTH}" \
            --prompt "${PROMPT}" \
            --ep_size "${EP_SIZE}" &

        PIDS+=($!)
    done

    echo "All ${MA_NUM_GPUS} processes launched, waiting..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" || echo "Process $pid exited with error"
    done
    echo "All processes finished"
fi