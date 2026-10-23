#!/bin/bash
# End-to-end L1<->L2 (D2H + H2D) roundtrip test for the layer_first AscendC
# (mla.py) HiCache path on NPU.
#
# Forces device-pool eviction (D2H -> L2 host pool) with a long filler request,
# then replays the evicted prefix to force an L2 hit + H2D restore, and asserts
# both kv_exchange_copy directions appear in the server log and the needle answer
# is preserved. See e2e_layer_first_roundtrip.py for details.
#
# Runs INSIDE the NPU container (sglang_lws_2).

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/DeepSeek-Coder-V2-Lite-Instruct}"
PORT="${PORT:-32345}"
HOST="${HOST:-127.0.0.1}"
NPU_DEVICES="${NPU_DEVICES:-8}"
BASE_URL="${BASE_URL:-http://$HOST:$PORT}"
LOG_DIR="${LOG_DIR:-/home/l00889655/sglang/e2e_layer_first_logs}"
NEEDLE="${NEEDLE:-48219}"
DOC_TOKENS="${DOC_TOKENS:-10000}"
EVICT_TOKENS="${EVICT_TOKENS:-14000}"

mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/roundtrip_server.log"

export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
# Keep the container's CANN site-packages (provides `acl`) on PYTHONPATH.
export PYTHONPATH="/home/l00889655/sglang/sglang/python:${PYTHONPATH:-}"

echo "=== E2E: layer_first MLA L1<->L2 roundtrip (D2H + H2D) ==="
echo "model:      $MODEL_PATH"
echo "base url:   $BASE_URL"
echo "server log: $SERVER_LOG"

# ---------------------------------------------------------------------------
# 1. Launch the server. host pool (ratio 3.0) is deliberately larger than the
#    device pool (max_total_tokens) so evicted L1 tokens are retained in L2.
# ---------------------------------------------------------------------------
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp 1 \
    --enable-hicache \
    --hicache-mem-layout layer_first \
    --hicache-io-backend kernel_ascend \
    --hicache-write-policy write_through \
    --hicache-ratio 2.5 \
    --mem-fraction-static 0.75 \
    --max-total-tokens 16384 \
    --device npu \
    --attention-backend ascend \
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
for i in $(seq 1 300); do
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
python "$(dirname "$0")/e2e_layer_first_roundtrip.py" \
    --base-url "$BASE_URL" \
    --server-log "$SERVER_LOG" \
    --needle "$NEEDLE" \
    --model-path "$MODEL_PATH" \
    --doc-tokens "$DOC_TOKENS" \
    --evict-tokens "$EVICT_TOKENS"

echo "done."
exit 0