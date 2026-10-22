#!/bin/bash
# End-to-end smoke test for the layer_first AscendC (mla.py) HiCache path on NPU.
#
# Validates that:
#   1. The server starts with --hicache-mem-layout layer_first + kernel_ascend
#      + SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric (no startup rejection).
#   2. A single request returns a coherent, correct completion (exercises the
#      L2 host pool build + D2H/H2D transfer path).
#
# Runs INSIDE the NPU container (sglang_lws_2), which has torch.npu,
# memfabric_hybrid and the model weights mounted.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/DeepSeek-Coder-V2-Lite-Instruct}"
PORT="${PORT:-32345}"
HOST="${HOST:-127.0.0.1}"
NPU_DEVICES="${NPU_DEVICES:-8}"
LOG_DIR="${LOG_DIR:-/home/l00889655/sglang/e2e_layer_first_logs}"

mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"
RESP_LOG="$LOG_DIR/response.json"

export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
# Prepend the local sglang checkout; MUST also keep the container's CANN
# site-packages (which supplies the `acl` NPU module) on PYTHONPATH. A bare
# override here silently breaks `deepseek_v2`/`kimi_k3`/`glm`/`mamba` imports
# and forces a Transformers backend fallback.
export PYTHONPATH="/home/l00889655/sglang/sglang/python:${PYTHONPATH:-}"

echo "=== E2E: layer_first MLA HiCache smoke test ==="
echo "model:        $MODEL_PATH"
echo "port:         $PORT"
echo "npu devices:  $NPU_DEVICES"
echo "server log:   $SERVER_LOG"

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
    --max-total-tokens 4096 \
    --log-level info \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "server pid:   $SERVER_PID"

cleanup() {
    echo "=== tear down server (pid $SERVER_PID) ==="
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    # Best-effort: kill any lingering scheduler/launch child processes.
    pkill -f "launch_server.*32345" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Wait for the server to become healthy (with timeout).
# ---------------------------------------------------------------------------
health_url="http://$HOST:$PORT/health"
echo "waiting for health at $health_url ..."
ready=0
for i in $(seq 1 240); do
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
# 3. Send a single request and verify a correct completion.
# ---------------------------------------------------------------------------
echo "sending request to /v1/completions ..."
curl -fsS "http://$HOST:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "DeepSeek-Coder-V2-Lite-Instruct",
        "prompt": "Question: what is 1 + 1? Answer with a single number.\nAnswer:",
        "max_tokens": 8,
        "temperature": 0.0
    }' > "$RESP_LOG"

echo "--- raw response ---"
cat "$RESP_LOG"
echo

# Extract the generated text via python (json parsing).
GENERATED="$(
    python - "$RESP_LOG" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
text = data["choices"][0]["text"]
print(text)
PYEOF
)"

echo "--- generated text ---"
echo "[$GENERATED]"

# Correctness check: the completion must be non-empty and contain "2".
if [ -z "$GENERATED" ]; then
    echo "FAIL: empty completion"
    exit 1
fi
if ! echo "$GENERATED" | grep -q "2"; then
    echo "FAIL: completion does not contain the expected answer '2': [$GENERATED]"
    exit 1
fi

echo "PASS: layer_first MLA HiCache smoke test succeeded (completion: [$GENERATED])"
exit 0