#!/usr/bin/env bash
# Dedicated test server only. Choose eight FREE physical devices explicitly.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
: "${LAYOUT:?Set layer_first or page_first}"
: "${NPU_DEVICES:?Set eight free physical NPU devices}"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
python3 - <<'PY'
import os
ids=os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')
assert len(ids)==8 and len(set(ids))==8 and all(x.isdigit() for x in ids), 'Provide eight distinct physical NPU IDs'
PY
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32 SGLANG_SET_CPU_AFFINITY=1
export HCCL_BUFFSIZE=2048 HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo HCCL_INTRA_DMA_ENABLE=1
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=True SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False SGLANG_OPT_FP8_WO_A_GEMM=False
export SGLANG_ENABLE_SPEC_V2=0 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0 SGLANG_NPU_USE_MULTI_STREAM=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128 DEEPEP_HCCL_BUFFSIZE=1024
export SGLANG_ZBAL_LOCAL_MEM_SIZE=0 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=1
export SGLANG_ENABLE_WAR_BARRIER=1 SGLANG_FORCE_COARSE_WAR_BARRIER=1 SGLANG_DSV4_AIV_TRACE=0
unset DSV4_GSM_AIV_AUDIT_DIR
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
export PYTHONPATH="${SGLANG_ROOT:-/home/l00889655/sglang/sglang}/python:${PYTHONPATH:-}"
exec python3 -u -m sglang.launch_server \
  --model-path "${MODEL_PATH:-/mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8}" \
  --host 127.0.0.1 --port "${PORT:-32420}" --trust-remote-code \
  --tp-size 8 --ep-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
  --device npu --attention-backend dsv4 --quantization modelslim --kv-cache-dtype bfloat16 \
  --page-size 128 --swa-full-tokens-ratio 0.1 --max-total-tokens 66000 \
  --max-running-requests 16 --chunked-prefill-size 32768 --mem-fraction-static 0.8 \
  --moe-a2a-backend deepep --deepep-mode auto --watchdog-timeout 9000 --random-seed 0 \
  --context-length 1048576 --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
  --enable-hierarchical-cache --hicache-mem-layout "$LAYOUT" \
  --hicache-io-backend kernel_ascend --hicache-write-policy write_through --hicache-ratio 2.5 --stream-interval 1 --log-level warning
