#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Required for two containers on the same host:
#   NODE0_IP: container/node0 communication IP, used as MASTER_ADDR
#   NODE1_IP: container/node1 communication IP
NODE0_IP="${NODE0_IP:-}"
NODE1_IP="${NODE1_IP:-${LOCAL_HOST:-}}"
if [ -z "$NODE0_IP" ] || [ -z "$NODE1_IP" ]; then
    echo "Please set NODE0_IP and NODE1_IP. Example:"
    echo "  NODE0_IP=172.17.0.2 NODE1_IP=172.17.0.3 $0 /data00/dpskv4-flash-quant"
    exit 1
fi

export MOJO_NODE_IPS="${NODE0_IP} ${NODE1_IP}"
export LOCAL_HOST="${LOCAL_HOST:-$NODE1_IP}"
export NODE_RANK=1

# Container/node1 uses the last 8 global NPU devices by default. If the
# container runtime already remaps visible devices to 0..7, set NPU_DEVICE_BASE=0.
export WORLD_SIZE="${WORLD_SIZE:-16}"
export EP_SIZE="${EP_SIZE:-16}"
export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-8}"
export NPU_DEVICE_BASE="${NPU_DEVICE_BASE:-0}"

export CP_SIZE="${CP_SIZE:-1}"
export ATTN_TP_SIZE="${ATTN_TP_SIZE:-1}"
export LMHEAD_TP_SIZE="${LMHEAD_TP_SIZE:-4}"
export O_PROJ_TP_SIZE="${O_PROJ_TP_SIZE:-1}"

export MASTER_ADDR="${MASTER_ADDR:-$NODE0_IP}"
export MASTER_PORT="${MASTER_PORT:-6328}"
export HCCL_IF_IP="${HCCL_IF_IP:-$NODE1_IP}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"

exec "${SCRIPT_DIR}/run_llm_multi_node.sh" "$@"
