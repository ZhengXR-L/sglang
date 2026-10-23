#!/bin/bash
# End-to-end layer_first (mla.py AscendC) D2H + H2D roundtrip test for the
# GLM-5.2-w4a8c8 DSA + Indexer pool on NPU, using the default bf16 KV-cache.
#
# Purpose: verify the layer_first Indexer (index_k) transport path end-to-end.
# GLM-5.2 is pure DSA (GlmMoeDsaForCausalLM, no mamba) and, on NPU with the
# ascend backend, routes through NPUMLATokenToKVPool -> mla.py. Its physical
# indexer layers (21 of 78) mean each AscendC kv_exchange_copy must assemble
# k + v + index_k (no index_scale under bf16; that is FP8-packing only), so the
# runner asserts the new "(k,v,index_k)" component marker in both directions.
#
# Reuses e2e_layer_first_roundtrip.py unchanged for the needle roundtrip.
#
# Runs INSIDE the NPU container (sglang-lzx-B103).

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/GLM-5.2-w4a8c8}"
PORT="${PORT:-32345}"
HOST="${HOST:-127.0.0.1}"
NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7}"
BASE_URL="${BASE_URL:-http://$HOST:$PORT}"
LOG_DIR="${LOG_DIR:-/home/l00889655/sglang/e2e_layer_first_glm52_logs}"
NEEDLE="${NEEDLE:-48219}"
DOC_TOKENS="${DOC_TOKENS:-2500}"
EVICT_TOKENS="${EVICT_TOKENS:-2800}"

mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/roundtrip_server.log"

export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
export HCCL_BUFFSIZE=512
# B103 does not put PyACL on PYTHONPATH by default; SGLang's GLM model
# import needs acl even before the server initializes its NPU workers.
export PYTHONPATH="/home/l00889655/sglang/sglang/python:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages:${PYTHONPATH:-}"
python -c "import acl; import sglang.srt.models.glm4_moe"

echo "=== E2E: layer_first GLM-5.2-w4a8c8 DSA+Indexer roundtrip (bf16) ==="
echo "model:      $MODEL_PATH"
echo "base url:   $BASE_URL"
echo "server log: $SERVER_LOG"

# ---------------------------------------------------------------------------
# 1. Launch the server. host pool (ratio 2.5) is larger than the device pool
#    (max_total_tokens) so evicted L1 tokens are retained in L2.
# ---------------------------------------------------------------------------
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp-size 8 \
    --ep 8 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --deepep-dispatcher-output-dtype bf16 \
    --enable-hicache \
    --hicache-mem-layout layer_first \
    --hicache-io-backend kernel_ascend \
    --hicache-write-policy write_through \
    --hicache-ratio 2.5 \
    --mem-fraction-static 0.86 \
    --max-total-tokens 3072 \
    --device npu \
    --attention-backend ascend \
    --disable-cuda-graph \
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
for i in $(seq 1 360); do
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
# 3. Run the needle roundtrip (W/D2H -> E/evict -> R/H2D) unchanged.
# ---------------------------------------------------------------------------
python "$(dirname "$0")/e2e_layer_first_roundtrip.py" \
    --base-url "$BASE_URL" \
    --server-log "$SERVER_LOG" \
    --needle "$NEEDLE" \
    --model-path "$MODEL_PATH" \
    --doc-tokens "$DOC_TOKENS" \
    --evict-tokens "$EVICT_TOKENS"
ROUNDTRIP_RC=$?
if [ "$ROUNDTRIP_RC" -ne 0 ]; then
    echo "FAIL: needle roundtrip failed (rc=$ROUNDTRIP_RC)"
    exit "$ROUNDTRIP_RC"
fi

# ---------------------------------------------------------------------------
# 4. Assert the Indexer (index_k) component marker in both directions.
#    bf16 GLM-5.2 DSA -> components = k + v + index_k (no index_scale).
#    The server log now emits "components=3 (k,v,index_k)" via mla.py.
# ---------------------------------------------------------------------------
IDX_D2H=$(grep -cF "kv_exchange_copy D2H" "$SERVER_LOG" || true)
IDX_H2D=$(grep -cF "kv_exchange_copy H2D" "$SERVER_LOG" || true)
IDX_COMPONENTS=$(grep -cF "(k,v,index_k)" "$SERVER_LOG" || true)

echo "indexer check: D2H=$IDX_D2H H2D=$IDX_H2D (k,v,index_k)_markers=$IDX_COMPONENTS"

if [ "$IDX_D2H" -eq 0 ] || [ "$IDX_H2D" -eq 0 ]; then
    echo "FAIL: missing kv_exchange_copy direction marker (D2H=$IDX_D2H H2D=$IDX_H2D)"
    exit 1
fi
if [ "$IDX_COMPONENTS" -eq 0 ]; then
    echo "FAIL: no (k,v,index_k) component marker — index_k was not transferred"
    exit 1
fi

echo "PASS: layer_first GLM-5.2-w4a8c8 Indexer (index_k) D2H+H2D roundtrip OK"
exit 0