#!/bin/bash
# End-to-end GSM8K accuracy test for the layer_first AscendC (mla.py) HiCache
# path on NPU.
#
# Validates that, with --hicache-mem-layout layer_first + kernel_ascend +
# SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric, the model still produces accurate
# GSM8K answers (i.e. the L2 host pool build + per-request D2H/H2D transfer does
# not corrupt generation). Runs 100 questions by default.
#
# Runs INSIDE the NPU container (sglang_lws_2).

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/DeepSeek-Coder-V2-Lite-Instruct}"
PORT="${PORT:-32345}"
HOST="${HOST:-127.0.0.1}"
NPU_DEVICES="${NPU_DEVICES:-8}"
NUM_QUESTIONS="${NUM_QUESTIONS:-100}"
NUM_SHOTS="${NUM_SHOTS:-5}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-16384}"
DATA_PATH="${DATA_PATH:-/home/l00889655/sglang/gsm8k_test.jsonl}"
BASE_URL="${BASE_URL:-http://$HOST:$PORT}"
LOG_DIR="${LOG_DIR:-/home/l00889655/sglang/e2e_layer_first_logs}"

mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/gsm8k_server.log"
EVAL_LOG="$LOG_DIR/gsm8k_eval.log"

export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
# Prepend the local sglang checkout; MUST keep the container's CANN
# site-packages (which supplies the `acl` NPU module) on PYTHONPATH.
export PYTHONPATH="/home/l00889655/sglang/sglang/python:${PYTHONPATH:-}"

echo "=== E2E: layer_first MLA GSM8K accuracy test ==="
echo "model:         $MODEL_PATH"
echo "base url:      $BASE_URL"
echo "num questions: $NUM_QUESTIONS (shots=$NUM_SHOTS)"
echo "server log:    $SERVER_LOG"
echo "eval log:      $EVAL_LOG"

# ---------------------------------------------------------------------------
# 1. Launch the server (background), forcing the layer_first + AscendC path.
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
    --hicache-ratio 0.5 \
    --mem-fraction-static 0.6 \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --log-level info \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "server pid:    $SERVER_PID"

cleanup() {
    echo "=== tear down server (pid $SERVER_PID) ==="
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    pkill -f "launch_server.*${PORT}" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Wait for the server to become healthy (with timeout).
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
# 3. Run the GSM8K evaluation.
# ---------------------------------------------------------------------------
echo "running GSM8K eval (${NUM_QUESTIONS} questions)..."
python "$(dirname "$0")/e2e_gsm8k_eval.py" \
    --base-url "$BASE_URL" \
    --data-path "$DATA_PATH" \
    --num-questions "$NUM_QUESTIONS" \
    --num-shots "$NUM_SHOTS" \
    2>&1 | tee "$EVAL_LOG"

echo "done."
exit 0