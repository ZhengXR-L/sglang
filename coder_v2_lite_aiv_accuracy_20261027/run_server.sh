#!/usr/bin/env bash
# Dedicated TP1 server. Select one idle physical NPU explicitly.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
: "${AUDIT_DIR:?Set a new empty AUDIT_DIR}"
: "${NPU_DEVICES:?Set one free physical NPU ID}"
[[ "$NPU_DEVICES" =~ ^[0-9]+$ ]] || { echo 'NPU_DEVICES must be one device' >&2; exit 2; }
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
export PYTHONPATH="${SGLANG_ROOT:-/home/l00889655/sglang/sglang}/python:${PYTHONPATH:-}"
exec python3 -u "$HERE/gsm8k_aiv.py" serve \
  --audit-dir "$AUDIT_DIR" --sglang-root "${SGLANG_ROOT:-/home/l00889655/sglang/sglang}" -- \
  --model-path "${MODEL_PATH:-/mnt/paas/weights/DeepSeek-Coder-V2-Lite-Instruct}" \
  --host 127.0.0.1 --port "${PORT:-32345}" --trust-remote-code \
  --tp 1 --device npu --attention-backend ascend \
  --enable-hicache --hicache-mem-layout layer_first \
  --hicache-io-backend kernel_ascend --hicache-write-policy write_through \
  --hicache-ratio 2.0 --mem-fraction-static 0.6 --max-total-tokens 16384 \
  --log-level info
