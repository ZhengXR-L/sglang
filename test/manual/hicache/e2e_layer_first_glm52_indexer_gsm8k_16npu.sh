#!/usr/bin/env bash
# GSM8K accuracy smoke test for GLM-5.2-w4a8c8 with NPU layer_first HiCache.
# Run inside sglang-lzx-B103. Uses the same serving configuration as the
# passing 16-NPU Indexer roundtrip test.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/GLM-5.2-w4a8c8}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-32348}"
NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
BASE_URL="${BASE_URL:-http://$HOST:$PORT}"
DATA_PATH="${DATA_PATH:-/home/l00889655/sglang/gsm8k_test.jsonl}"
NUM_QUESTIONS="${NUM_QUESTIONS:-100}"
NUM_SHOTS="${NUM_SHOTS:-5}"
MAX_TOKENS="${MAX_TOKENS:-512}"
PARALLEL="${PARALLEL:-16}"
LOG_DIR="${LOG_DIR:-/home/l00889655/sglang/e2e_layer_first_glm52_gsm8k_16npu_logs}"

mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/gsm8k_server.log"
EVAL_LOG="$LOG_DIR/gsm8k_eval.log"

export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
export HCCL_BUFFSIZE=1024
export PYTHONPATH="/home/l00889655/sglang/sglang/python:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages:${PYTHONPATH:-}"
python -c "import acl; import sglang.srt.models.glm4_moe"

echo "=== GLM-5.2-w4a8c8 layer_first GSM8K, TP16/EP16 ==="
echo "model:      $MODEL_PATH"
echo "base url:   $BASE_URL"
echo "questions:  $NUM_QUESTIONS (shots=$NUM_SHOTS, max_tokens=$MAX_TOKENS, parallel=$PARALLEL)"
echo "server log: $SERVER_LOG"
echo "eval log:   $EVAL_LOG"

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp-size 16 \
    --ep 16 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --deepep-dispatcher-output-dtype bf16 \
    --enable-hicache \
    --hicache-mem-layout layer_first \
    --hicache-io-backend kernel_ascend \
    --hicache-write-policy write_through \
    --hicache-ratio 2.5 \
    --mem-fraction-static 0.85 \
    --max-total-tokens 8192 \
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

health_url="http://$HOST:$PORT/health"
ready=0
for i in $(seq 1 360); do
    if curl -fsS "$health_url" >/dev/null 2>&1; then
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

python "$(dirname "$0")/e2e_gsm8k_eval.py" \
    --base-url "$BASE_URL" \
    --data-path "$DATA_PATH" \
    --num-questions "$NUM_QUESTIONS" \
    --num-shots "$NUM_SHOTS" \
    --max-tokens "$MAX_TOKENS" \
    --parallel "$PARALLEL" \
    2>&1 | tee "$EVAL_LOG"

echo "GSM8K evaluation complete."
