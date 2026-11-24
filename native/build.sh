#!/usr/bin/env bash
set -eo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: bash native/build.sh <provider> [operator]\nProvider is an architecture or architecture/sku_<name>.\nExamples: npu_a2 varlen_fa; npu_a5/sku_950pr swa\nSet MOJO_NATIVE_KEEP_BUILD=1 to retain a successful build directory.\n'
    exit 0
fi
if [[ "$#" -lt 1 || "$#" -gt 2 || ! "$1" =~ ^[a-z0-9_]+(/sku_[a-z0-9_]+)?$ || ! "${2:-all}" =~ ^[a-z0-9_]+$ ]]; then
    printf 'Expected: bash native/build.sh <provider> [operator]\n' >&2
    exit 2
fi
NATIVE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${NATIVE_DIR}/.." && pwd)"
PROVIDER="$1"
OPERATOR="${2:-all}"
PLATFORM="${PROVIDER%%_*}"
if [[ ! -d "${NATIVE_DIR}/src/${PROVIDER}" || ! -f "${NATIVE_DIR}/cmake/${PLATFORM}.cmake" ||
      ( "${OPERATOR}" != all && ! -f "${NATIVE_DIR}/src/${PROVIDER}/${OPERATOR}/CMakeLists.txt" ) ]]; then
    printf 'Expected: an existing native provider/operator recipe\n' >&2
    exit 2
fi
BUILD_ROOT="${REPO_DIR}/build/native/${PROVIDER}/${OPERATOR}"
KEEP_BUILD="${MOJO_NATIVE_KEEP_BUILD:-0}"
if [[ "${KEEP_BUILD}" != 0 && "${KEEP_BUILD}" != 1 ]]; then
    printf 'MOJO_NATIVE_KEEP_BUILD must be 0 or 1\n' >&2
    exit 2
fi
if [[ "${PLATFORM}" == npu && -z "${ASCEND_HOME_PATH:-}" ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
set -u
# CANN rewrites objects during preprocessing; each build owns a fresh tree.
mkdir -p "${BUILD_ROOT}"
BUILD_DIR="$(mktemp -d "${BUILD_ROOT}/build.XXXXXX")"
cleanup_build() {
    local build_status=$?
    if [[ "${build_status}" == 0 && "${KEEP_BUILD}" == 0 ]]; then
        # Remove only the exact temporary directory created by this invocation.
        rm -rf -- "${BUILD_DIR}"
    else
        printf 'Retained native build directory: %s\n' "${BUILD_DIR}" >&2
    fi
}
trap cleanup_build EXIT
printf 'Native build directory: %s\n' "${BUILD_DIR}"
NATIVE_CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=Release"
    "-DMOJO_NATIVE_PROVIDER=${PROVIDER}"
    "-DMOJO_NATIVE_OP=${OPERATOR}"
    "-DPython3_EXECUTABLE=$(command -v python)"
)
if [[ "${PLATFORM}" == npu ]]; then
    NATIVE_CMAKE_ARGS+=("-DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}")
    if [[ -n "${SOC_VERSION:-}" ]]; then
        NATIVE_CMAKE_ARGS+=("-DSOC_VERSION=${SOC_VERSION}")
    fi
    if [[ -n "${CATLASS_ROOT:-}" ]]; then
        NATIVE_CMAKE_ARGS+=("-DCATLASS_ROOT=${CATLASS_ROOT}")
    fi
fi
cmake -S "${NATIVE_DIR}" -B "${BUILD_DIR}" "${NATIVE_CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --target mojo_native -j"${BUILD_JOBS:-8}"
