#!/bin/bash

# Source CANN environment
CANN_PATH="${CANN_PATH:-/usr/local/Ascend/cann}"
if [ -f "${CANN_PATH}/bin/setenv.bash" ]; then
    source "${CANN_PATH}/bin/setenv.bash"
fi
if [ -f "${CANN_PATH}/opp/vendors/customize/bin/set_env.bash" ]; then
    source "${CANN_PATH}/opp/vendors/customize/bin/set_env.bash"
fi
if [ -f "${CANN_PATH}/opp/vendors/custom_transformer/bin/set_env.bash" ]; then
    source "${CANN_PATH}/opp/vendors/custom_transformer/bin/set_env.bash"
fi
rm -rf /tmp/torchinductor_root
# export MOJO_GRAPH_CACHE_DIR=/data01/tbw/mojo_opset_info/compile_cache_smoke_1780105319
# rm -rf ${MOJO_GRAPH_CACHE_DIR}
# Ensure /usr/local/lib64 is searched first for GLIBCXX compatibility
export LD_LIBRARY_PATH="/usr/local/lib64:${LD_LIBRARY_PATH:-}"

set -euo pipefail
env
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DEFAULT_LOCAL_PATH="/data00/dpskv4-flash-quant"
export MOJO_BACKEND="ascendc"
export MOJO_GRAPH_MODE="${MOJO_GRAPH_MODE:-npugraph_ex}"
export MOJO_PROF="${MOJO_PROF:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
# export PROMPT="${PROMPT:-[\"请用一句话介绍量子计算的核心原理。\"]}"
export PROMPT="${PROMPT:-[\"请用一句话介绍量子计算的核心原理，并说明它与经典计算最关键的差异，回答要简洁清晰，适合高中生理解，不要超过两句话，重点突出叠加态和纠缠。\", \"请用通俗但准确的语言系统介绍量子计算的核心原理。请先解释量子比特与经典比特的区别，说明叠加态如何让一个量子系统同时保留多种可能性。再说明纠缠为什么能建立远距离量子相关，以及测量为什么会把概率结果转化为确定输出。接着解释量子门和量子线路如何改变状态幅度，并通过干涉放大正确答案、抑制错误答案。请用一个简单例子说明量子算法为什么可能在特定问题上加速，例如搜索、分子模拟或大整数分解。最后补充当前工程挑战，包括退相干、量子纠错、门保真度、可扩展控制、低温系统和实际应用落地。整体避免复杂公式，但概念要严谨，适合有基础计算机知识的读者阅读，并给出清晰总结。请强调量子计算不是简单替代经典计算，而是在少数结构化任务上可能展现优势。请强调量子计算不是简单替代经典计算，而是在少数结构化任务上可能展现优势。请强调量子计算不是简单替代经典计算，而是在少数结构化任务上可能展现优势。请强调量子计算不是简单替代经典计算，而是在少数结构化任务上可能展现优势。请强调量子计算不是简单替代经典计算，而是在少数结构化任务上可能展现优势。请强调量子计算不是简单替代经典计算，而是在少数结构化任务上可能展现优势。请强调量子计算不是简单替代经典计算，而是在少\"]}"
# export PROMPT="${PROMPT:-[\"请用一句话介绍量子计算的核心原理。请用一句话介绍量子计算的核心原理。请用一句话介绍量子计算的核心原理。请用一句话介绍量子计算的核心原理。请用一句话介绍量子计算的核心原理。请用一句话介绍量子计算的核心原理。请用一句话介绍量子计算的核心原理。请用一句话介绍量子计算的核心原理。请用一句话介绍量子计算的核心原理。请用一句话介绍量子计算的核心原理。请用一句话介绍量子计算的核心原理。请用一句话介绍量子计算的核心原理。\"]}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}"
    exit 1
fi

cleanup_stale_processes() {
    if [ "${MOJO_CLEAN_STALE_PROCS:-1}" != "1" ]; then
        echo "Skip stale process cleanup: MOJO_CLEAN_STALE_PROCS=${MOJO_CLEAN_STALE_PROCS}"
        return 0
    fi

    local patterns=(
        "${PROJECT_ROOT}/examples/run_llm.sh"
        "${PROJECT_ROOT}/examples/llm_inference.py"
        "python3 -m examples.llm_inference"
        "python -m examples.llm_inference"
    )
    local current_pid="$$"
    local stale_pids=""
    local pat pid cmd

    for pat in "${patterns[@]}"; do
        while IFS= read -r pid; do
            [ -n "${pid}" ] || continue
            [ "${pid}" = "${current_pid}" ] && continue
            cmd="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
            [ -n "${cmd}" ] || continue
            case "${cmd}" in
                *pgrep*|*cleanup_stale_processes*) continue ;;
            esac
            stale_pids="${stale_pids} ${pid}"
        done < <(pgrep -f "${pat}" || true)
    done

    stale_pids="$(echo "${stale_pids}" | tr ' ' '\n' | awk 'NF && !seen[$0]++' | tr '\n' ' ')"
    if [ -n "${stale_pids// }" ]; then
        echo "Killing stale inference processes:${stale_pids}"
        kill -TERM ${stale_pids} 2>/dev/null || true
        sleep 2
        kill -KILL ${stale_pids} 2>/dev/null || true
    else
        echo "No stale inference processes found"
    fi
}

cleanup_child_processes() {
    if [ "${MOJO_CLEAN_ON_EXIT:-1}" != "1" ]; then
        return 0
    fi
    if [ "${#PIDS[@]}" -eq 0 ]; then
        return 0
    fi

    local live_pids=()
    local pid
    for pid in "${PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            live_pids+=("${pid}")
        fi
    done
    if [ "${#live_pids[@]}" -gt 0 ]; then
        echo "Cleaning child inference processes: ${live_pids[*]}"
        kill -TERM "${live_pids[@]}" 2>/dev/null || true
        sleep 2
        kill -KILL "${live_pids[@]}" 2>/dev/null || true
    fi
}

PIDS=()
# trap cleanup_child_processes EXIT INT TERM

MODEL_PATH="${1:-$DEFAULT_LOCAL_PATH}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at ${MODEL_PATH}"
    exit 1
fi

echo "Running inference with model at: ${MODEL_PATH}"
# cleanup_stale_processes

if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
    echo "Current python cannot import torch: ${PYTHON_BIN}"
    exit 1
fi

export MOJO_BACKEND="${MOJO_BACKEND:-ascendc}"
export MOJO_DISABLE_ASSERTION_REWRITE="${MOJO_DISABLE_ASSERTION_REWRITE:-1}"
export MOJO_MOE_MULTI_STREAM="${MOJO_MOE_MULTI_STREAM:-1}"
export MOJO_ATTN_MLA_MULTI_STREAM="${MOJO_ATTN_MLA_MULTI_STREAM:-1}"
export MOJO_ATTN_COMPRESSOR_MULTI_STREAM="${MOJO_ATTN_COMPRESSOR_MULTI_STREAM:-1}"
export MOJO_ATTN_INDEXER_MULTI_STREAM="${MOJO_ATTN_INDEXER_MULTI_STREAM:-1}"
export MOJO_SKIP_GUARD_EVAL_AFTER_WARMUP="${MOJO_SKIP_GUARD_EVAL_AFTER_WARMUP:-1}"

EP_SIZE="${EP_SIZE:-8}"
CP_SIZE="${CP_SIZE:-${EP_SIZE}}"
ATTN_TP_SIZE="${ATTN_TP_SIZE:-1}"
LMHEAD_TP_SIZE="${LMHEAD_TP_SIZE:-4}"
O_PROJ_TP_SIZE="${O_PROJ_TP_SIZE:-1}"
MOJO_USE_PARALLELIZE_MODULE_TP="${MOJO_USE_PARALLELIZE_MODULE_TP:-0}"
MOJO_USE_PARALLELIZE_MODULE_EP="${MOJO_USE_PARALLELIZE_MODULE_EP:-0}"
# npugraph_ex static kernel needs LOCAL_WORLD_SIZE in multi-card launches.
export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-${EP_SIZE}}"
NUM_LAYERS="${LLM_NUM_LAYERS:-43}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
PA_MAX_LENGTH="${PA_MAX_LENGTH:-4096}"
INPUT_MAX_LEN="${INPUT_MAX_LEN:-0}"
NEXT_N="${NEXT_N:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
export USE_ATTN_METADATA=1
USE_ATTN_METADATA="${MOJO_USE_ATTN_METADATA:-1}"
if [ "${MOJO_BUILD_LEGACY_ATTN_INPUTS:-}" = "" ]; then
    if [ "${USE_ATTN_METADATA}" = "1" ]; then
        export MOJO_BUILD_LEGACY_ATTN_INPUTS="0"
    else
        export MOJO_BUILD_LEGACY_ATTN_INPUTS="1"
    fi
fi

cd "$PROJECT_ROOT" || exit 1

if [ "$EP_SIZE" -eq 1 ]; then
    echo "EP=1, CP=${CP_SIZE}, ATTN_TP=${ATTN_TP_SIZE}, LMHEAD_TP=${LMHEAD_TP_SIZE}, O_PROJ_TP=${O_PROJ_TP_SIZE}, use_parallelize_module_tp=${MOJO_USE_PARALLELIZE_MODULE_TP}, use_parallelize_module_ep=${MOJO_USE_PARALLELIZE_MODULE_EP}, batch_size=${BATCH_SIZE}, next_n=${NEXT_N}, use_attn_metadata=${USE_ATTN_METADATA}, build_legacy_attn_inputs=${MOJO_BUILD_LEGACY_ATTN_INPUTS}, moe_multi_stream=${MOJO_MOE_MULTI_STREAM}, attn_mla_multi_stream=${MOJO_ATTN_MLA_MULTI_STREAM}, attn_compressor_multi_stream=${MOJO_ATTN_COMPRESSOR_MULTI_STREAM}, attn_indexer_multi_stream=${MOJO_ATTN_INDEXER_MULTI_STREAM}, skip_guard_eval_after_warmup=${MOJO_SKIP_GUARD_EVAL_AFTER_WARMUP}"
    ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}" \
    "${PYTHON_BIN}" -m examples.llm_inference \
        --model_path "${MODEL_PATH}" \
        --device "${DEVICE:-npu}" \
        --num_layers "${NUM_LAYERS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --pa_max_length "${PA_MAX_LENGTH}" \
        --input_max_len "${INPUT_MAX_LEN}" \
        --next_n "${NEXT_N}" \
        --prompt "${PROMPT}" \
        --ep_size 1 \
        --cp_size "${CP_SIZE}" \
        --attn_tp_size "${ATTN_TP_SIZE}" \
        --lmhead_tp_size "${LMHEAD_TP_SIZE}" \
        --o_proj_tp_size "${O_PROJ_TP_SIZE}" \
        --batch_size "${BATCH_SIZE}" \
        --use_attn_metadata "${USE_ATTN_METADATA}" \
        --next_n "${NEXT_N}" \
        --use_parallelize_module_tp "${MOJO_USE_PARALLELIZE_MODULE_TP}" \
        --use_parallelize_module_ep "${MOJO_USE_PARALLELIZE_MODULE_EP}"
else
    echo "EP=${EP_SIZE}, CP=${CP_SIZE}, ATTN_TP=${ATTN_TP_SIZE}, LMHEAD_TP=${LMHEAD_TP_SIZE}, O_PROJ_TP=${O_PROJ_TP_SIZE}, use_parallelize_module_tp=${MOJO_USE_PARALLELIZE_MODULE_TP}, use_parallelize_module_ep=${MOJO_USE_PARALLELIZE_MODULE_EP}, multi-card inference, batch_size=${BATCH_SIZE}, next_n=${NEXT_N}, use_attn_metadata=${USE_ATTN_METADATA}, build_legacy_attn_inputs=${MOJO_BUILD_LEGACY_ATTN_INPUTS}, moe_multi_stream=${MOJO_MOE_MULTI_STREAM}, attn_mla_multi_stream=${MOJO_ATTN_MLA_MULTI_STREAM}, attn_compressor_multi_stream=${MOJO_ATTN_COMPRESSOR_MULTI_STREAM}, attn_indexer_multi_stream=${MOJO_ATTN_INDEXER_MULTI_STREAM}, skip_guard_eval_after_warmup=${MOJO_SKIP_GUARD_EVAL_AFTER_WARMUP}"

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
            --input_max_len "${INPUT_MAX_LEN}" \
            --next_n "${NEXT_N}" \
            --prompt "${PROMPT}" \
            --ep_size "${EP_SIZE}" \
            --cp_size "${CP_SIZE}" \
            --attn_tp_size "${ATTN_TP_SIZE}" \
            --lmhead_tp_size "${LMHEAD_TP_SIZE}" \
            --o_proj_tp_size "${O_PROJ_TP_SIZE}" \
            --batch_size "${BATCH_SIZE}" \
            --use_attn_metadata "${USE_ATTN_METADATA}" \
            --next_n "${NEXT_N}" \
            --use_parallelize_module_tp "${MOJO_USE_PARALLELIZE_MODULE_TP}" \
            --use_parallelize_module_ep "${MOJO_USE_PARALLELIZE_MODULE_EP}" &

        PIDS+=($!)
    done

    echo "All ${MA_NUM_GPUS} processes launched, waiting..."
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "Process $pid exited with error"
            # cleanup_child_processes
            exit 1
        fi
    done
    echo "All processes finished"
fi
