#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
: "${LAYOUT:?}"
: "${NPU_DEVICES:?}"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
export PYTHONPATH="${SGLANG_ROOT:-/home/l00889655/sglang/sglang}/python:${PYTHONPATH:-}"
exec python3 -u -m sglang.launch_server \
  --model-path /mnt/paas/weights/DeepSeek-Coder-V2-Lite-Instruct \
  --host 127.0.0.1 --port "${PORT:-32431}" --trust-remote-code \
  --tp 1 --device npu --attention-backend ascend \
  --enable-hicache --hicache-mem-layout "$LAYOUT" \
  --hicache-io-backend kernel_ascend --hicache-write-policy write_through \
  --hicache-ratio 2.0 --mem-fraction-static 0.6 --max-total-tokens 10240 \
  --chunked-prefill-size 2048 \
  --stream-interval 1 --log-level info
