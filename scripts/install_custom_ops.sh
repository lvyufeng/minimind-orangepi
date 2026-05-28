#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUSTOM_OPP_INSTALL="${ASCEND_CUSTOM_OPP_PATH:-${ROOT_DIR}/custom_opp_install}"

if [[ -z "${ASCEND_TOOLKIT_ROOT:-${ASCEND_HOME_PATH:-}}" ]]; then
  echo "ASCEND_TOOLKIT_ROOT or ASCEND_HOME_PATH is not set. Source scripts/set_env.sh first." >&2
  exit 1
fi

if [[ ! -d "${ROOT_DIR}/src/csrc/kernels/custom_ops" ]]; then
  echo "No custom AscendC ops are defined yet."
  echo "Expected future custom op source at: ${ROOT_DIR}/src/csrc/kernels/custom_ops"
  echo "Install destination would be: ${CUSTOM_OPP_INSTALL}"
  exit 0
fi

echo "Custom op sources exist, but installer wiring has not been implemented yet."
echo "Install destination: ${CUSTOM_OPP_INSTALL}"
