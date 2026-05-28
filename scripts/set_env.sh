#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${ASCEND_TOOLKIT_ROOT:-}" ]]; then
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    export ASCEND_TOOLKIT_ROOT="${ASCEND_HOME_PATH}"
  elif [[ -d /usr/local/Ascend/cann/latest ]]; then
    export ASCEND_TOOLKIT_ROOT=/usr/local/Ascend/cann/latest
  elif [[ -d /usr/local/Ascend/cann ]]; then
    export ASCEND_TOOLKIT_ROOT=/usr/local/Ascend/cann
  elif [[ -d /usr/local/Ascend/cann-8.5.0 ]]; then
    export ASCEND_TOOLKIT_ROOT=/usr/local/Ascend/cann-8.5.0
  fi
fi

if [[ -n "${ASCEND_TOOLKIT_ROOT:-}" ]]; then
  export ASCEND_HOME_PATH="${ASCEND_TOOLKIT_ROOT}"
  export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-${ASCEND_TOOLKIT_ROOT}/opp}"
  export ASCEND_AICPU_PATH="${ASCEND_AICPU_PATH:-${ASCEND_TOOLKIT_ROOT}}"
  export TOOLCHAIN_HOME="${TOOLCHAIN_HOME:-${ASCEND_TOOLKIT_ROOT}/toolkit}"

  for lib_dir in \
    "${ASCEND_TOOLKIT_ROOT}/lib64" \
    "${ASCEND_TOOLKIT_ROOT}/aarch64-linux/lib64" \
    "${ASCEND_TOOLKIT_ROOT}/x86_64-linux/lib64"; do
    if [[ -d "${lib_dir}" ]]; then
      export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
    fi
  done
fi

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-${ROOT_DIR}/custom_opp_install}"
export PYTHONPATH="${ROOT_DIR}/src/python:${PYTHONPATH:-}"

echo "ROOT_DIR=${ROOT_DIR}"
echo "ASCEND_TOOLKIT_ROOT=${ASCEND_TOOLKIT_ROOT:-}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-}"
echo "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
echo "PYTHONPATH=${PYTHONPATH}"
