#!/usr/bin/env bash
set -eo pipefail
: "${LAYOUT:?}"
: "${NPU_DEVICES:?}"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
export HCCL_BUFFSIZE=512
export PYTHONPATH="/home/l00889655/sglang/sglang/python:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages:${PYTHONPATH:-}"
python3 -c 'import acl; import sglang.srt.models.glm4_moe'
exec python3 -u -m sglang.launch_server \
  --model-path /mnt/paas/weights/GLM-5.2-w4a8c8 \
  --host 127.0.0.1 --port "${PORT:-32432}" \
  --tp-size 8 --ep 8 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --deepep-dispatcher-output-dtype bf16 \
  --enable-hicache --hicache-mem-layout "$LAYOUT" \
  --hicache-io-backend kernel_ascend --hicache-write-policy write_through \
  --hicache-ratio 2.5 --mem-fraction-static 0.86 --max-total-tokens 3072 \
  --device npu --attention-backend ascend --disable-cuda-graph --log-level info
