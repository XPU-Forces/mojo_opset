#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_LOCAL_PATH="/data00/dpskv4-flash-quant"

# Source CANN environment.
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

export LD_LIBRARY_PATH="/usr/local/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

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

# Golden-style multi-node topology. For the first migration step we target EP16.
WORLD_SIZE="${WORLD_SIZE:-16}"
EP_SIZE="${EP_SIZE:-16}"
LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-${MOJO_LOCAL_WORLD_SIZE:-8}}"
NPU_DEVICE_BASE="${NPU_DEVICE_BASE:-0}"
MASTER_PORT="${MASTER_PORT:-6038}"
HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-23456}"
HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1200}"
HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"

if [ "$WORLD_SIZE" -ne "$EP_SIZE" ]; then
    echo "EP16 migration currently expects WORLD_SIZE == EP_SIZE, got WORLD_SIZE=${WORLD_SIZE}, EP_SIZE=${EP_SIZE}"
    exit 1
fi
if [ $((WORLD_SIZE % LOCAL_WORLD_SIZE)) -ne 0 ]; then
    echo "WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE, got WORLD_SIZE=${WORLD_SIZE}, LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
    exit 1
fi

# IP list can be provided either as MOJO_NODE_IPS="ip0 ip1" or IPs bash array
# before sourcing/executing this script. The first IP is the master address.
if [ -n "${MOJO_NODE_IPS:-}" ]; then
    read -r -a IP_ARRAY <<< "${MOJO_NODE_IPS}"
elif declare -p IPs >/dev/null 2>&1; then
    # shellcheck disable=SC2154
    IP_ARRAY=("${IPs[@]}")
else
    LOCAL_HOST_DEFAULT="$(hostname -I | awk '{print $1}')"
    IP_ARRAY=("${MASTER_ADDR:-$LOCAL_HOST_DEFAULT}")
fi

SERVER_NUM=$(((WORLD_SIZE + LOCAL_WORLD_SIZE - 1) / LOCAL_WORLD_SIZE))
if [ "${#IP_ARRAY[@]}" -lt "$SERVER_NUM" ]; then
    echo "Need ${SERVER_NUM} node IPs for WORLD_SIZE=${WORLD_SIZE}, LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}, got ${#IP_ARRAY[@]}: ${IP_ARRAY[*]}"
    exit 1
fi
IP_ARRAY=("${IP_ARRAY[@]:0:$SERVER_NUM}")

LOCAL_HOST="${LOCAL_HOST:-$(hostname -I | awk '{print $1}')}"
NODE_RANK="${NODE_RANK:-}"
if [ -z "$NODE_RANK" ]; then
    NODE_RANK=-1
    for idx in "${!IP_ARRAY[@]}"; do
        if [ "$LOCAL_HOST" = "${IP_ARRAY[$idx]}" ]; then
            NODE_RANK="$idx"
            break
        fi
    done
fi
if [ "$NODE_RANK" -lt 0 ] || [ "$NODE_RANK" -ge "$SERVER_NUM" ]; then
    echo "Cannot determine valid NODE_RANK. LOCAL_HOST=${LOCAL_HOST}, IPs=(${IP_ARRAY[*]}), NODE_RANK=${NODE_RANK}"
    echo "Set NODE_RANK explicitly if hostname -I does not match MOJO_NODE_IPS."
    exit 1
fi

RANK_OFFSET=$((NODE_RANK * LOCAL_WORLD_SIZE))
RANKS_LEFT=$((WORLD_SIZE - RANK_OFFSET))
if [ "$RANKS_LEFT" -le 0 ]; then
    echo "No ranks assigned to this node: NODE_RANK=${NODE_RANK}, RANK_OFFSET=${RANK_OFFSET}, WORLD_SIZE=${WORLD_SIZE}"
    exit 1
fi
LOCAL_RANKS_TO_LAUNCH="$LOCAL_WORLD_SIZE"
if [ "$RANKS_LEFT" -lt "$LOCAL_RANKS_TO_LAUNCH" ]; then
    LOCAL_RANKS_TO_LAUNCH="$RANKS_LEFT"
fi

export WORLD_SIZE
export EP_SIZE
export LOCAL_WORLD_SIZE="$LOCAL_RANKS_TO_LAUNCH"
export RANK_OFFSET
export MASTER_ADDR="${MASTER_ADDR:-${IP_ARRAY[0]}}"
export MASTER_PORT
export HCCL_SOCKET_IFNAME
export HCCL_IF_IP="${HCCL_IF_IP:-$LOCAL_HOST}"
export HCCL_IF_BASE_PORT
export HCCL_CONNECT_TIMEOUT
export HCCL_EXEC_TIMEOUT
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

export MOJO_BACKEND="${MOJO_BACKEND:-ascendc}"
export MOJO_GRAPH_MODE="${MOJO_GRAPH_MODE:-npugraph_ex}"
export MOJO_PROF="${MOJO_PROF:-0}"
export MOJO_DISABLE_ASSERTION_REWRITE="${MOJO_DISABLE_ASSERTION_REWRITE:-1}"
export MOJO_MOE_MULTI_STREAM="${MOJO_MOE_MULTI_STREAM:-1}"
export MOJO_ATTN_MLA_MULTI_STREAM="${MOJO_ATTN_MLA_MULTI_STREAM:-1}"
export MOJO_ATTN_COMPRESSOR_MULTI_STREAM="${MOJO_ATTN_COMPRESSOR_MULTI_STREAM:-1}"
export MOJO_ATTN_INDEXER_MULTI_STREAM="${MOJO_ATTN_INDEXER_MULTI_STREAM:-1}"
export MOJO_SKIP_GUARD_EVAL_AFTER_WARMUP="${MOJO_SKIP_GUARD_EVAL_AFTER_WARMUP:-1}"
export USE_ATTN_METADATA=1
USE_ATTN_METADATA="${MOJO_USE_ATTN_METADATA:-1}"
if [ "${MOJO_BUILD_LEGACY_ATTN_INPUTS:-}" = "" ]; then
    if [ "$USE_ATTN_METADATA" = "1" ]; then
        export MOJO_BUILD_LEGACY_ATTN_INPUTS="0"
    else
        export MOJO_BUILD_LEGACY_ATTN_INPUTS="1"
    fi
fi

NUM_LAYERS="${LLM_NUM_LAYERS:-43}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PA_MAX_LENGTH="${PA_MAX_LENGTH:-2048}"
NEXT_N="${NEXT_N:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CP_SIZE="${CP_SIZE:-1}"
ATTN_TP_SIZE="${ATTN_TP_SIZE:-1}"
LMHEAD_TP_SIZE="${LMHEAD_TP_SIZE:-4}"
O_PROJ_TP_SIZE="${O_PROJ_TP_SIZE:-1}"
PROMPT="${PROMPT:-[\"请用一句话介绍量子计算的核心原理。\"]}"

if [ $((WORLD_SIZE % LMHEAD_TP_SIZE)) -ne 0 ]; then
    echo "WORLD_SIZE must be divisible by LMHEAD_TP_SIZE, got WORLD_SIZE=${WORLD_SIZE}, LMHEAD_TP_SIZE=${LMHEAD_TP_SIZE}"
    exit 1
fi

DATE_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${MOJO_LOG_ROOT:-${PROJECT_ROOT}/logs/multi_node_ep${EP_SIZE}_w${WORLD_SIZE}_${DATE_TAG}}"
mkdir -p "$LOG_ROOT"

if [ -z "${MOJO_GRAPH_CACHE_DIR:-}" ]; then
    export MOJO_GRAPH_CACHE_DIR="${PROJECT_ROOT}/compile_cache/ep${EP_SIZE}_w${WORLD_SIZE}_b${BATCH_SIZE}_n${NEXT_N}_cp${CP_SIZE}_lm${LMHEAD_TP_SIZE}"
fi
mkdir -p "$MOJO_GRAPH_CACHE_DIR"

echo "Running multi-node inference with model at: ${MODEL_PATH}"
echo "Topology: WORLD_SIZE=${WORLD_SIZE}, EP_SIZE=${EP_SIZE}, SERVER_NUM=${SERVER_NUM}, NODE_RANK=${NODE_RANK}, RANK_OFFSET=${RANK_OFFSET}, LOCAL_RANKS=${LOCAL_RANKS_TO_LAUNCH}"
echo "Device mapping: NPU_DEVICE_BASE=${NPU_DEVICE_BASE}, local ranks use npu:${NPU_DEVICE_BASE}..npu:$((NPU_DEVICE_BASE + LOCAL_RANKS_TO_LAUNCH - 1))"
echo "IPs=(${IP_ARRAY[*]}), MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"
echo "HCCL_IF_IP=${HCCL_IF_IP}, HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}, HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT}"
echo "Parallel: CP_SIZE=${CP_SIZE}, ATTN_TP_SIZE=${ATTN_TP_SIZE}, LMHEAD_TP_SIZE=${LMHEAD_TP_SIZE}, O_PROJ_TP_SIZE=${O_PROJ_TP_SIZE}"
echo "Runtime: BATCH_SIZE=${BATCH_SIZE}, MAX_NEW_TOKENS=${MAX_NEW_TOKENS}, PA_MAX_LENGTH=${PA_MAX_LENGTH}, NEXT_N=${NEXT_N}, GRAPH_CACHE=${MOJO_GRAPH_CACHE_DIR}"
echo "Streams: MOE=${MOJO_MOE_MULTI_STREAM}, MLA=${MOJO_ATTN_MLA_MULTI_STREAM}, COMPRESSOR=${MOJO_ATTN_COMPRESSOR_MULTI_STREAM}, INDEXER=${MOJO_ATTN_INDEXER_MULTI_STREAM}, SKIP_GUARD_AFTER_WARMUP=${MOJO_SKIP_GUARD_EVAL_AFTER_WARMUP}"
echo "Logs: ${LOG_ROOT}"

cd "$PROJECT_ROOT"

PIDS=()
cleanup_child_processes() {
    if [ "${#PIDS[@]}" -eq 0 ]; then
        return 0
    fi
    local live_pids=()
    local pid
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            live_pids+=("$pid")
        fi
    done
    if [ "${#live_pids[@]}" -gt 0 ]; then
        echo "Cleaning child inference processes: ${live_pids[*]}"
        kill -TERM "${live_pids[@]}" 2>/dev/null || true
        sleep 2
        kill -KILL "${live_pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup_child_processes INT TERM

for ((local_rank=0; local_rank<LOCAL_RANKS_TO_LAUNCH; local_rank++)); do
    export LOCAL_RANK="$local_rank"
    export RANK_ID=$((RANK_OFFSET + local_rank))
    export NPU_DEVICE_IDX=$((NPU_DEVICE_BASE + local_rank))

    cmd=(
        "$PYTHON_BIN" -m examples.llm_inference
        --model_path "$MODEL_PATH"
        --device "${DEVICE:-npu}"
        --num_layers "$NUM_LAYERS"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --pa_max_length "$PA_MAX_LENGTH"
        --next_n "$NEXT_N"
        --prompt "$PROMPT"
        --ep_size "$EP_SIZE"
        --cp_size "$CP_SIZE"
        --attn_tp_size "$ATTN_TP_SIZE"
        --lmhead_tp_size "$LMHEAD_TP_SIZE"
        --o_proj_tp_size "$O_PROJ_TP_SIZE"
        --batch_size "$BATCH_SIZE"
        --use_attn_metadata "$USE_ATTN_METADATA"
    )

    echo "Launching rank ${RANK_ID} (local_rank=${LOCAL_RANK}, npu_device=${NPU_DEVICE_IDX})"
    if [ "${MOJO_DRY_RUN:-0}" = "1" ]; then
        printf 'DRY_RUN rank %s command:' "$RANK_ID"
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
    fi

    "${cmd[@]}" > "${LOG_ROOT}/log_${LOCAL_RANK}_rank${RANK_ID}.log" 2>&1 &
    PIDS+=("$!")
done

if [ "${MOJO_DRY_RUN:-0}" = "1" ]; then
    echo "Dry run finished."
    exit 0
fi

echo "All ${#PIDS[@]} local processes launched, waiting..."
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "Process $pid exited with error. Check logs under ${LOG_ROOT}"
        cleanup_child_processes
        exit 1
    fi
done

echo "All local processes finished. Logs: ${LOG_ROOT}"
