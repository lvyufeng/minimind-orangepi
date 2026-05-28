#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
ASCEND_TOOLKIT_ROOT="${ASCEND_TOOLKIT_ROOT:-${ASCEND_HOME_PATH:-}}"

cmake_args=(
  -S "${ROOT_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
)

if [[ -n "${ASCEND_TOOLKIT_ROOT}" ]]; then
  cmake_args+=("-DMINIMIND_ASCEND_TOOLKIT_ROOT=${ASCEND_TOOLKIT_ROOT}")
fi

cmake "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" --parallel "${JOBS:-$(nproc)}"
