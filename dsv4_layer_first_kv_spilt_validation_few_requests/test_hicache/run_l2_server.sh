#!/bin/bash
set -eo pipefail
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

SGL_SRC_HOME=${SGL_SRC_HOME:-/home/cx/sglang}
LOG_DIR=${LOG_DIR:-/home/cx/log/dsv4_hicache_$(date +%Y%m%d_%H%M%S)}
export MODEL_PATH=${MODEL_PATH:-/mnt/weight/DeepSeek-V4-Flash-DSpark}
export PORT=${PORT:-30000}
DP_RANKS=${DP_RANKS:-8}
mkdir -p "$LOG_DIR"
SERVER_LOG=${SERVER_LOG:-$LOG_DIR/server.log}
SERVER_PID_FILE=${SERVER_PID_FILE:-$LOG_DIR/server.pid}

# Refuse to attach the test to an unrelated existing server on this port.
python3 - <<'PY'
import os, socket
with socket.socket() as sock:
    sock.bind(("0.0.0.0", int(os.environ["PORT"])))
PY

export PYTHONPATH="$SGL_SRC_HOME/python:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=${NPU_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1
export HCCL_BUFFSIZE=2048
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_INTRA_DMA_ENABLE=1
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False
export SGLANG_HICACHE_HOST_MEM_BACKEND=${SGLANG_HICACHE_HOST_MEM_BACKEND:-memfabric}
export SGLANG_ENABLE_SPEC_V2=${SGLANG_ENABLE_SPEC_V2:-1}
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=100
export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1

args=(--model-path "$MODEL_PATH" --page-size 128
      --tp-size "${TP_SIZE:-8}" --ep-size "${EP_SIZE:-1}"
      --trust-remote-code --attention-backend dsv4 --device npu
      --watchdog-timeout 9000 --host 0.0.0.0 --port "$PORT"
      --mem-fraction-static "${MEM_FRACTION_STATIC:-0.72}"
      --max-running-requests 16 --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-32768}"
      --max-total-tokens "${MAX_TOTAL_TOKENS:-66000}"
      --quantization "${QUANTIZATION:-fp8}" --reasoning-parser deepseek-v4
      --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}"
      --random-seed "${RANDOM_SEED:-20260824}" --context-length 1048576
      --dp-size "$DP_RANKS" --enable-hierarchical-cache
      --hicache-io-backend kernel_ascend --hicache-ratio "${HICACHE_RATIO:-2.0}"
      --hicache-write-policy write_through
      --hicache-mem-layout "${HICACHE_LAYOUT:-page_first}")
if (( DP_RANKS > 1 )); then args+=(--enable-dp-attention --enable-dp-lm-head); fi
if [[ "${ENABLE_MTP:-1}" == 1 ]]; then
    args+=(--speculative-algorithm EAGLE --speculative-num-steps 2
           --speculative-eagle-topk 1 --speculative-num-draft-tokens 3)
fi
if [[ "${DISABLE_GRAPHS:-0}" == 1 ]]; then
    args+=(--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled)
fi
if [[ -n "${MOE_A2A_BACKEND:-}" ]]; then
    args+=(--moe-a2a-backend "$MOE_A2A_BACKEND" --deepep-mode auto)
fi
if [[ -n "${SWA_FULL_TOKENS_RATIO:-}" ]]; then args+=(--swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO"); fi

git -C "$SGL_SRC_HOME" rev-parse HEAD > "$LOG_DIR/sglang_commit"
git -C "$SGL_SRC_HOME" status --short --branch > "$LOG_DIR/sglang_status.txt"
printf '%q ' python3 -m sglang.launch_server "${args[@]}" > "$LOG_DIR/launch_command.txt"
printf '\n' >> "$LOG_DIR/launch_command.txt"
python3 - <<'PY' > "$LOG_DIR/runtime_environment.json"
import hashlib, json, os
from pathlib import Path
import sglang, sgl_kernel_npu
keys = ["MODEL_PATH", "ASCEND_RT_VISIBLE_DEVICES", "PYTHONPATH", "HCCL_BUFFSIZE", "HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "HCCL_INTRA_DMA_ENABLE", "STREAMS_PER_DEVICE", "SGLANG_SET_CPU_AFFINITY", "PYTORCH_NPU_ALLOC_CONF", "SGLANG_DSV4_FP4_EXPERTS", "SGLANG_OPT_USE_OVERLAP_STORE_CACHE", "SGLANG_ENABLE_OVERLAP_PLAN_STREAM", "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "SGLANG_HICACHE_HOST_MEM_BACKEND"]
module = Path(sgl_kernel_npu.__file__).resolve()
lib = module.parent / "lib/libsgl_kernel_npu.so"
print(json.dumps({"environment": {k: os.environ.get(k) for k in keys}, "sglang_module": sglang.__file__, "kernel_module": str(module), "kernel_sha256": hashlib.sha256(lib.read_bytes()).hexdigest()}, indent=2))
PY
python3 -m sglang.launch_server "${args[@]}" > "$SERVER_LOG" 2>&1 &
echo "$!" > "$SERVER_PID_FILE"
echo "[INFO] started PID=$(cat "$SERVER_PID_FILE"); log=$SERVER_LOG"
