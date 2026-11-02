#!/usr/bin/env bash
# Dedicated TP8/EP8 GLM-5.2 layer-first KV-split server.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ASCEND_TOOLKIT_ROOT="${ASCEND_TOOLKIT_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
CANN_PYTHON_SITE_PACKAGES="${CANN_PYTHON_SITE_PACKAGES:-/usr/local/Ascend/cann-9.1.0/python/site-packages}"
ATB_ENV_SCRIPT="${ATB_ENV_SCRIPT:-/usr/local/Ascend/nnal/atb/set_env.sh}"
source "$ASCEND_TOOLKIT_ROOT/opp/vendors/customize/bin/set_env.bash"
source "$ASCEND_TOOLKIT_ROOT/opp/vendors/custom_transformer/bin/set_env.bash"
source "$ASCEND_TOOLKIT_ROOT/set_env.sh"
source "$ATB_ENV_SCRIPT"
set -u
: "${AUDIT_DIR:?Set a fresh AUDIT_DIR}"
: "${NPU_DEVICES:?Set eight free physical NPU IDs}"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
python3 - <<'PY'
import os
ids=os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')
assert len(ids)==8 and len(set(ids))==8 and all(x.isdigit() for x in ids)
PY
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
SGLANG_ROOT="${SGLANG_ROOT:-/home/l00889655/sglang/sglang}"
KERNEL_ROOT="${KERNEL_ROOT:-/home/l00889655/sglang/sgl-kernel-npu}"
test "$(git -C "$SGLANG_ROOT" branch --show-current)" = cjz_op
test "$(git -C "$KERNEL_ROOT" branch --show-current)" = npu/hicache-sparse-copy
export HCCL_BUFFSIZE=512
export PYTHONPATH="$SGLANG_ROOT/python:$KERNEL_ROOT/python/sgl_kernel_npu:$ASCEND_TOOLKIT_ROOT/python/site-packages:$CANN_PYTHON_SITE_PACKAGES:${PYTHONPATH:-}"
python3 -c 'import acl; import sglang.srt.models.glm4_moe; import sgl_kernel_npu.operators.mem_cache.kv_transfer as op; print("KV transfer operators:", op.__file__); assert all(hasattr(op, x) for x in ("transfer_kv_all_layer_kv_split", "transfer_kv_per_layer_kv_split"))'
exec python3 -u "$HERE/gsm8k_aiv.py" serve \
  --audit-dir "$AUDIT_DIR" --sglang-root "$SGLANG_ROOT" -- \
  --model-path "${MODEL_PATH:-/mnt/paas/weights/GLM-5.2-w4a8c8}" \
  --host 127.0.0.1 --port "${PORT:-32348}" \
  --tp-size 8 --ep 8 --moe-a2a-backend deepep --deepep-mode normal \
  --deepep-dispatcher-output-dtype bf16 \
  --enable-hicache --hicache-mem-layout layer_first \
  --hicache-io-backend kernel_ascend --hicache-write-policy write_through \
  --hicache-ratio 2.5 --mem-fraction-static 0.86 --max-total-tokens 3072 \
  --device npu --attention-backend ascend --disable-cuda-graph --log-level info
