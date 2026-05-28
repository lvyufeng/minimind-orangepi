#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${1:-${ROOT_DIR}/models/minimind}"
RUNTIME_DIR="${2:-${ROOT_DIR}/models/minimind-runtime}"
PROMPT="${PROMPT:-你好}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BENCH_TOKENS="${BENCH_TOKENS:-64}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

if ! find "${CHECKPOINT_DIR}" -maxdepth 1 \( -name '*.safetensors' -o -name 'pytorch_model.bin' -o -name '*.pth' \) | grep -q .; then
  echo "No MiniMind checkpoint weights found in ${CHECKPOINT_DIR}" >&2
  echo "Place model.safetensors, pytorch_model.bin, or a .pth checkpoint there first." >&2
  exit 1
fi

if [[ -f "${CHECKPOINT_DIR}/tokenizer.json" ]]; then
  if ! python3 - <<'PY' >/dev/null 2>&1
import tokenizers
PY
  then
    echo "tokenizer.json is present, but the Python 'tokenizers' package is not installed." >&2
    echo "Install it to decode prompts/text via run_text.py, or use build/minimind_generate --tokens directly." >&2
    exit 1
  fi
fi

"${ROOT_DIR}/scripts/build.sh"
python3 "${ROOT_DIR}/src/python/tools/inspect_minimind_checkpoint.py" "${CHECKPOINT_DIR}"
python3 "${ROOT_DIR}/src/python/tools/export_minimind_runtime.py" --model "${CHECKPOINT_DIR}" --output "${RUNTIME_DIR}"
python3 "${ROOT_DIR}/src/python/tools/run_text.py" --model "${RUNTIME_DIR}" --prompt "${PROMPT}" --max-new-tokens "${MAX_NEW_TOKENS}"
"${ROOT_DIR}/build/bench_decode" --model "${RUNTIME_DIR}" --prompt "${PROMPT}" --tokens "${BENCH_TOKENS}" --warmup 1
