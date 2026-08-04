#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

BASE_PYTHONPATH="${SCRIPT_DIR}:${PROJECT_ROOT}"

export NPU_PROF_MODE=${NPU_PROF_MODE:-}
export NPU_PROF_DIR=${NPU_PROF_DIR:-"${SCRIPT_DIR}/prof_dir"}

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"${SCRIPT_DIR}/inductor_cache"}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-12}
export TORCHNPU_PRECOMPILE_THREADS=${TORCHNPU_PRECOMPILE_THREADS:-12}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

PYTHON_BIN=${PYTHON_BIN:-python3}
rm -rf "${TORCHINDUCTOR_CACHE_DIR}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

usage() {
    cat <<'EOF'
Usage:
  run.sh [mask_mod|all] [seqlen|all] [--dry-run] [test_flex_attn.py args...]

mask_mod:
  sparse_mask_mod, sparse, full_mask_mod, full, all

seqlen:
  1k, 2k, 4k, 8k, 16k, 32k, 52k, 64k, 131k, all
  Multiple values can be comma-separated, for example: 1k,2k,52k

Examples:
  ./run.sh
  ./run.sh sparse_mask_mod 1k
  ./run.sh sparse_mask_mod 1k,2k --target triton
  ./run.sh all 52k
  ./run.sh all all --iters 5
  ./run.sh all all --dry-run

Default selection is sparse_mask_mod 1k,131k,52k.
By default this runner adds --perf-only so long seqlen cases compare triton and
inductor without building the SDPA reference. Set FLEXATTENTION_PERF_ONLY=0 to
run the correctness path.
EOF
}

normalize_mask() {
    local value="${1,,}"
    case "${value}" in
        all) echo "all" ;;
        sparse|sparse_mask_mod) echo "sparse_mask_mod" ;;
        full|full_mask_mod) echo "full_mask_mod" ;;
        *) return 1 ;;
    esac
}

normalize_seqlen() {
    local value="${1,,}"
    case "${value}" in
        all) echo "all" ;;
        1k|1000) echo "1k" ;;
        2k|2000) echo "2k" ;;
        4k|4000) echo "4k" ;;
        8k|8000) echo "8k" ;;
        16k|16000) echo "16k" ;;
        32k|32000) echo "32k" ;;
        52k|52000) echo "52k" ;;
        64k|64000) echo "64k" ;;
        131k|131000) echo "131k" ;;
        *) return 1 ;;
    esac
}

normalize_csv_filter() {
    local kind="$1"
    local raw="${2// /}"
    local normalized=()
    local item
    IFS=',' read -r -a items <<< "${raw}"
    for item in "${items[@]}"; do
        if [[ -z "${item}" ]]; then
            continue
        fi
        if [[ "${kind}" == "mask" ]]; then
            item="$(normalize_mask "${item}")" || {
                echo "invalid mask_mod: ${item}" >&2
                return 1
            }
        else
            item="$(normalize_seqlen "${item}")" || {
                echo "invalid seqlen: ${item}" >&2
                return 1
            }
        fi
        normalized+=("${item}")
    done
    if [[ "${#normalized[@]}" -eq 0 ]]; then
        echo "empty ${kind} filter" >&2
        return 1
    fi
    local joined
    printf -v joined ",%s" "${normalized[@]}"
    echo "${joined},"
}

filter_contains() {
    local filter="$1"
    local value="$2"
    [[ "${filter}" == *",all,"* || "${filter}" == *",${value},"* ]]
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

mask_filter="sparse_mask_mod"
seqlen_filter="1k,131k,52k"
if [[ $# -gt 0 && "${1}" != --* ]]; then
    mask_filter="$1"
    shift
fi
if [[ $# -gt 0 && "${1}" != --* ]]; then
    seqlen_filter="$1"
    shift
fi

mask_filter="$(normalize_csv_filter mask "${mask_filter}")" || {
    usage >&2
    exit 2
}
seqlen_filter="$(normalize_csv_filter seqlen "${seqlen_filter}")" || {
    usage >&2
    exit 2
}

extra_args=("$@")
dry_run=0
remaining_args=()
for arg in "${extra_args[@]}"; do
    if [[ "${arg}" == "--dry-run" ]]; then
        dry_run=1
    else
        remaining_args+=("${arg}")
    fi
done
extra_args=("${remaining_args[@]}")
default_args=()
if [[ "${FLEXATTENTION_PERF_ONLY:-1}" != "0" ]]; then
    default_args+=(--perf-only)
fi

CASES=(
    'sparse_mask_mod|1k|1000|[[90,800,110]]'
    'sparse_mask_mod|131k|131000|[[1500,6000,1500],[1300,8400,1300],[1100,10800,1100],[900,13200,900],[700,15600,700],[600,17800,600],[800,21400,800],[600,22800,600]]'
    'sparse_mask_mod|64k|64000|[[1200,5600,1200],[1000,7000,1000],[800,8400,800],[600,9800,600],[700,10600,700],[700,12600,700]]'
    'sparse_mask_mod|32k|64000|[[1200,5600,1200],[1000,7000,1000],[800,8400,800],[600,9800,600],[700,10600,700],[700,12600,700]]'
    'sparse_mask_mod|16k|16000|[[500,3000,500],[600,3800,600],[800,5400,800]]'
    'sparse_mask_mod|8k|8000|[[400,2200,400],[600,3800,600]]'
    'sparse_mask_mod|4k|4000|[[500,3000,500]]'
    'sparse_mask_mod|2k|2000|[[90,1400,510]]'
    'sparse_mask_mod|52k|52000|[[2000,22000,2000],[2000,22000,2000]]'
    'full_mask_mod|52k|52000|[[2000,22000,2000],[2000,22000,2000]]'
)

selected=0
for case_spec in "${CASES[@]}"; do
    IFS='|' read -r case_mask case_seqlen case_total case_data_length <<< "${case_spec}"
    if ! filter_contains "${mask_filter}" "${case_mask}"; then
        continue
    fi
    if ! filter_contains "${seqlen_filter}" "${case_seqlen}"; then
        continue
    fi
    selected=$((selected + 1))
    echo
    echo "=== flexattention case ${selected}: mask_mod=${case_mask}, seqlen=${case_seqlen}, total_s=${case_total} ==="
    echo "data_length=${case_data_length}"
    if [[ "${dry_run}" -eq 1 ]]; then
        printf '%q ' "${PYTHON_BIN}" "${SCRIPT_DIR}/test_complex_flexattention.py" \
            "${case_mask}" \
            --data-length "${case_data_length}" \
            "${default_args[@]}" \
            "${extra_args[@]}"
        echo
        continue
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/test_complex_flexattention.py" \
        "${case_mask}" \
        --data-length "${case_data_length}" \
        "${default_args[@]}" \
        "${extra_args[@]}"
done

if [[ "${selected}" -eq 0 ]]; then
    echo "no test cases matched mask_mod/seqlen filters" >&2
    exit 2
fi
