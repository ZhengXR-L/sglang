#!/bin/bash
# End-to-end L1<->L2 (D2H + H2D) roundtrip test for the DeepSeek-V4 layer_first
# HiCache path on NPU (DeepSeekV4PagedHostPool / DeepSeekV4StateHostPool via
# transfer_kv_dim_exchange, memory_pool_host.py).
#
# Loads DeepSeek-V4-Flash-0731-w8a8 with the SAME quant/parallelism flags as the
# known-good GSM8K test (test_npu_cp_decode_attn_tp.py): --quantization modelslim,
# --kv-cache-dtype bfloat16, --attention-backend dsv4, --ep-size 8 + deepep, and
# the DSv4 env toggles SGLANG_DSV4_FP4_EXPERTS=False / SGLANG_OPT_FUSE_WQA_WKV=0.
#
# Runs INSIDE the NPU container (sglang-lzx-B103).

set -eo pipefail

# DSv4 uses the DSA sparse-attention vendor ops (aclnnSparseAttnSharedkvMetadata)
# and the HcPre op (aclnnHcPre). These live in vendor op_api libs, NOT the base
# libopapi.so; without sourcing these set_env scripts the runtime dlopen falls
# back to libopapi.so and fails with "not in libopapi.so" (see run_cp8.sh).
source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer/bin/set_env.bash

MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8}"
PORT="${PORT:-32346}"
HOST="${HOST:-127.0.0.1}"
NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,7,8,9,10}"
TP_SIZE="${TP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-8}"
BASE_URL="${BASE_URL:-http://$HOST:$PORT}"
LOG_DIR="${LOG_DIR:-/home/l00889655/sglang/e2e_layer_first_logs}"
NEEDLE="${NEEDLE:-48219}"
DOC_TOKENS="${DOC_TOKENS:-4096}"
EVICT_TOKENS="${EVICT_TOKENS:-4608}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-1.0}"
HICACHE_LAYOUT="${HICACHE_LAYOUT:-layer_first}"

mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/roundtrip_dsv4_server.log"

export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
# Put the CANN/ascend-toolkit site-packages (provides `acl`) and our edited
# sglang source on PYTHONPATH.
export PYTHONPATH="/home/l00889655/sglang/sglang/python:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages:/home/sgl-workspace/sglang/python:${PYTHONPATH:-}"

# --- DSv4 w8a8 load/serve env toggles (mirror TEST_ENVS of the GSM8K test) ---
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export STREAMS_PER_DEVICE=32
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_BUFFSIZE=1024
export DEEPEP_HCCL_BUFFSIZE=1024
export SGLANG_ZBAL_LOCAL_MEM_SIZE=0
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=1
export SGLANG_NPU_USE_MULTI_STREAM=0
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False

echo "=== E2E: layer_first DSv4 L1<->L2 roundtrip (D2H + H2D) ==="
echo "model:      $MODEL_PATH"
echo "devices:    $NPU_DEVICES  (tp=$TP_SIZE ep=$EP_SIZE)"
echo "base url:   $BASE_URL"
echo "server log: $SERVER_LOG"

if [ "${ENABLE_HICACHE:-1}" = "1" ]; then
    hicache_args=(
        --enable-hicache
        --hicache-mem-layout "$HICACHE_LAYOUT"
        --hicache-io-backend kernel_ascend
        --hicache-write-policy write_through
        --hicache-ratio 2.5
    )
else
    if [ "${DISABLE_RADIX_CACHE:-1}" = "1" ]; then
        hicache_args=(--disable-radix-cache)
    else
        hicache_args=()
    fi
fi

# ---------------------------------------------------------------------------
# 1. Launch the server. host pool (ratio) is larger than the device pool
#    (--max-total-tokens) so evicted L1 tokens are retained in L2.
# ---------------------------------------------------------------------------
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp-size "$TP_SIZE" \
    --dp-size 1 \
    --ep-size "$EP_SIZE" \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --quantization modelslim \
    --kv-cache-dtype bfloat16 \
    --page-size 128 \
    --swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO" \
    --mem-fraction-static 0.8 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --watchdog-timeout 600 \
    --random-seed 0 \
    --cuda-graph-backend-prefill disabled \
    --cuda-graph-backend-decode disabled \
    "${hicache_args[@]}" \
    --max-total-tokens 8192 \
    --log-level info \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "server pid: $SERVER_PID"

cleanup() {
    echo "=== tear down server (pid $SERVER_PID) ==="
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    pkill -f "launch_server.*${PORT}" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Wait for the server to become healthy.
# ---------------------------------------------------------------------------
health_url="http://$HOST:$PORT/health"
echo "waiting for health at $health_url ..."
ready=0
for i in $(seq 1 1800); do
    if curl -fsS "$health_url" > /dev/null 2>&1; then
        ready=1
        echo "server healthy after ~${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FATAL: server process exited early. See $SERVER_LOG"
        tail -80 "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done

if [ "$ready" -ne 1 ]; then
    echo "FATAL: server did not become healthy in time. See $SERVER_LOG"
    tail -120 "$SERVER_LOG"
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Run the roundtrip.
# ---------------------------------------------------------------------------
python "$(dirname "$0")/e2e_layer_first_dsv4_roundtrip.py" \
    --base-url "$BASE_URL" \
    --server-log "$SERVER_LOG" \
    --needle "$NEEDLE" \
    --model-path "$MODEL_PATH" \
    --doc-tokens "$DOC_TOKENS" \
    --evict-tokens "$EVICT_TOKENS"

echo "done."
exit 0
