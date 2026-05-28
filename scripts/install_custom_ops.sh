#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUSTOM_OP_SRC="${ROOT_DIR}/src/csrc/kernels/custom_ops"
CUSTOM_OP_BUILD="${ROOT_DIR}/build/custom_ops"
CUSTOM_OPP_INSTALL="${ASCEND_CUSTOM_OPP_PATH:-${ROOT_DIR}/custom_opp_install}"

ASCEND_TOOLKIT_ROOT="${ASCEND_TOOLKIT_ROOT:-}"
if [[ -z "${ASCEND_TOOLKIT_ROOT}" && -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]]; then
  ASCEND_TOOLKIT_ROOT="${ASCEND_HOME_PATH}"
fi
if [[ -z "${ASCEND_TOOLKIT_ROOT}" ]]; then
  for candidate in /usr/local/Ascend/cann/latest /usr/local/Ascend/cann /usr/local/Ascend/cann-8.5.0; do
    if [[ -d "${candidate}" ]]; then
      ASCEND_TOOLKIT_ROOT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${ASCEND_TOOLKIT_ROOT}" || ! -d "${ASCEND_TOOLKIT_ROOT}" ]]; then
  echo "Ascend CANN toolkit was not found. Source scripts/set_env.sh or set ASCEND_TOOLKIT_ROOT." >&2
  exit 1
fi

export ASCEND_TOOLKIT_ROOT
export ASCEND_HOME_PATH="${ASCEND_TOOLKIT_ROOT}"
export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-${ASCEND_TOOLKIT_ROOT}/opp}"
export ASCEND_AICPU_PATH="${ASCEND_AICPU_PATH:-${ASCEND_TOOLKIT_ROOT}}"
export TOOLCHAIN_HOME="${TOOLCHAIN_HOME:-${ASCEND_TOOLKIT_ROOT}/toolkit}"
for bin_dir in \
  "${ASCEND_TOOLKIT_ROOT}/bin" \
  "${ASCEND_TOOLKIT_ROOT}/aarch64-linux/bin" \
  "${ASCEND_TOOLKIT_ROOT}/x86_64-linux/bin" \
  "${ASCEND_TOOLKIT_ROOT}/tools/bisheng_compiler/bin"; do
  if [[ -d "${bin_dir}" ]]; then
    export PATH="${bin_dir}:${PATH}"
  fi
done
for lib_dir in \
  "${ASCEND_TOOLKIT_ROOT}/lib64" \
  "${ASCEND_TOOLKIT_ROOT}/aarch64-linux/lib64" \
  "${ASCEND_TOOLKIT_ROOT}/x86_64-linux/lib64" \
  "${ASCEND_TOOLKIT_ROOT}/tools/msprofiler/lib" \
  "${ASCEND_TOOLKIT_ROOT}/tools/mspti/lib64" \
  "${ASCEND_TOOLKIT_ROOT}/tools/bisheng_compiler/lib"; do
  if [[ -d "${lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
  fi
done

TEMPLATE_DIR="${ASCEND_TOOLKIT_ROOT}/tools/op_project_templates/ascendc/customize"
if [[ ! -d "${TEMPLATE_DIR}" ]]; then
  echo "AscendC custom op template was not found: ${TEMPLATE_DIR}" >&2
  exit 1
fi
if [[ ! -d "${CUSTOM_OP_SRC}/op_host" || ! -d "${CUSTOM_OP_SRC}/op_kernel" ]]; then
  echo "Custom op sources are missing under ${CUSTOM_OP_SRC}." >&2
  exit 1
fi

rm -rf "${CUSTOM_OP_BUILD}"
mkdir -p "${CUSTOM_OP_BUILD}"
cp -aL "${TEMPLATE_DIR}/." "${CUSTOM_OP_BUILD}/"
find "${CUSTOM_OP_BUILD}/op_host" -maxdepth 1 -type f ! -name CMakeLists.txt -delete
find "${CUSTOM_OP_BUILD}/op_kernel" -maxdepth 1 -type f -delete
cp -a "${CUSTOM_OP_SRC}/op_host/." "${CUSTOM_OP_BUILD}/op_host/"
cp -a "${CUSTOM_OP_SRC}/op_kernel/." "${CUSTOM_OP_BUILD}/op_kernel/"

python3 - <<PY
import json
from pathlib import Path
path = Path(r"${CUSTOM_OP_BUILD}/CMakePresets.json")
data = json.loads(path.read_text())
vars = data["configurePresets"][0]["cacheVariables"]
vars["ASCEND_COMPUTE_UNIT"]["value"] = "ascend310b"
vars["ASCEND_CANN_PACKAGE_PATH"]["value"] = r"${ASCEND_TOOLKIT_ROOT}"
vars["vendor_name"]["value"] = "minimind_orangepi"
vars["ENABLE_TEST"]["value"] = "False"
path.write_text(json.dumps(data, indent=4) + "\n")
PY

cmake -S "${CUSTOM_OP_BUILD}" -B "${CUSTOM_OP_BUILD}/build_out" --preset=default
cmake --build "${CUSTOM_OP_BUILD}/build_out" --target binary -j"$(nproc)"
cmake --build "${CUSTOM_OP_BUILD}/build_out" --target install -j"$(nproc)"

mkdir -p "${CUSTOM_OPP_INSTALL}"
if [[ -d "${CUSTOM_OP_BUILD}/build_out/packages" ]]; then
  rm -rf "${CUSTOM_OPP_INSTALL}/vendors/minimind_orangepi"
  cp -a "${CUSTOM_OP_BUILD}/build_out/packages/." "${CUSTOM_OPP_INSTALL}/"
else
  echo "Custom op package output was not generated." >&2
  exit 1
fi

echo "Installed MiniMind custom AscendC ops to: ${CUSTOM_OPP_INSTALL}"
