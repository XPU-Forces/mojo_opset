#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DEFAULT_LOCAL_PATH="/data00/DeepSeek-V4-Flash-INT8"
export MOJO_BACKEND="ascendc"
export MOJO_GRAPH_MODE="${MOJO_GRAPH_MODE:-npugraph_ex}"
export MOJO_PROF="0"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

export PROMPT="${PROMPT:-[\"请用一句话介绍量子计算的核心原理。\", \"在昇腾NPU平台上部署大语言模型进行推理时，需要综合考虑多个技术维度的优化策略。首先，在并行策略方面，张量并行适用于低时延场景，专家并行适用于高吞吐的MoE架构模型，数据并行则用于提升整体吞吐量。对于DeepSeek这 类MoE模型，通常采用专家并行加张量并行的混合并行策略，其中专家并行用于专家层的分布式计算，张量并行用于注意力层和共享层的切分。其次，在算子优化方面，FlashAttention可以显著降低注意力计算的显存占用和计算延迟，而融合算子则能减少算子间的访存开销和任务调度开销。第三，在显存管理方面，KVCache的优化至关重要，包括PagedAttention的块管理策略和KV缓存的量化压缩。第四，在图模式优化方面，torch.compile和GE图模式可以将动态图转换为静态图，消除Python解释器开销。第五，在多流并行方面，可以将 注意力计算和前馈网络计算分别放到不同的NPU流上执行，实现流水线重叠。第六，在权重预取方面，可以在计算当前层的同时预取下一层的权重，隐藏访存延迟。请基于以上背景 ，详细分析在昇腾NPU上部署DeepSeek模型时应该如何选择和组合这些优化技术。\"]}"

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

export MOJO_BACKEND="${MOJO_BACKEND:-ascendc}"
export MOJO_DISABLE_ASSERTION_REWRITE="${MOJO_DISABLE_ASSERTION_REWRITE:-1}"

EP_SIZE="${EP_SIZE:-8}"
NUM_LAYERS="${LLM_NUM_LAYERS:-43}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
PA_MAX_LENGTH="${PA_MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-2}"

cd "$PROJECT_ROOT" || exit 1

if [ "$EP_SIZE" -eq 1 ]; then
    echo "EP=1, single card inference, batch_size=${BATCH_SIZE}"
    ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}" \
    "${PYTHON_BIN}" -m examples.llm_inference \
        --model_path "${MODEL_PATH}" \
        --device "${DEVICE:-npu}" \
        --num_layers "${NUM_LAYERS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --pa_max_length "${PA_MAX_LENGTH}" \
        --prompt "${PROMPT}" \
        --ep_size 1 \
        --batch_size "${BATCH_SIZE}"
else
    echo "EP=${EP_SIZE}, multi-card inference, batch_size=${BATCH_SIZE}"

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
            --ep_size "${EP_SIZE}" \
            --batch_size "${BATCH_SIZE}" &

        PIDS+=($!)
    done

    echo "All ${MA_NUM_GPUS} processes launched, waiting..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" || echo "Process $pid exited with error"
    done
    echo "All processes finished"
fi