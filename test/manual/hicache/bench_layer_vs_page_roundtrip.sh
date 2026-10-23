#!/bin/bash
# HiCache layer_first vs page_first_kv_split: TTFT + throughput benchmark.
#
# Runs the three-round (fill -> evict -> replay) benchmark twice, once per
# --hicache-mem-layout, on the SAME model / IO backend / host-pool sizing, so
# the only variable is the host KV memory layout.
#
# Runs INSIDE the NPU container (e.g. sglang-lzx-B103).

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/DeepSeek-Coder-V2-Lite-Instruct}"
HOST="${HOST:-127.0.0.1}"
NPU_DEVICES="${NPU_DEVICES:-8}"
LOG_DIR="${LOG_DIR:-/home/l00889655/sglang/e2e_layer_first_logs}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1500}"
MAX_TOKENS="${MAX_TOKENS:-64}"

mkdir -p "$LOG_DIR"

export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
# Keep the container's CANN site-packages (provides `acl`) on PYTHONPATH.
export PYTHONPATH="/home/l00889655/sglang/sglang/python:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

run_one() {
    local layout="$1"
    local port="$2"
    local server_log="$LOG_DIR/bench_${layout}_server.log"
    local base_url="http://$HOST:$port"

    echo ""
    echo "===== layout=$layout (port=$port) ====="
    echo "server log: $server_log"

    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host "$HOST" \
        --port "$port" \
        --tp 1 \
        --enable-hicache \
        --hicache-mem-layout "$layout" \
        --hicache-io-backend kernel_ascend \
        --hicache-write-policy write_through \
        --hicache-ratio 2.5 \
        --mem-fraction-static 0.75 \
        --max-total-tokens 16384 \
        --log-level info \
        > "$server_log" 2>&1 &

    local server_pid=$!
    echo "server pid: $server_pid"

    # Wait for health.
    local ready=0
    for i in $(seq 1 300); do
        if curl -fsS "http://$HOST:$port/health" > /dev/null 2>&1; then
            ready=1
            echo "server healthy after ~${i}s"
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "FATAL: server exited early. See $server_log"
            tail -80 "$server_log"
            exit 1
        fi
        sleep 1
    done

    if [ "$ready" -ne 1 ]; then
        echo "FATAL: server did not become healthy. See $server_log"
        tail -120 "$server_log"
        exit 1
    fi

    # Run the benchmark.
    python "$SCRIPT_DIR/bench_layer_vs_page_roundtrip.py" \
        --base-url "$base_url" \
        --layout "$layout" \
        --server-log "$server_log" \
        --model-path "$MODEL_PATH" \
        --num-prompts "$NUM_PROMPTS" \
        --prompt-tokens "$PROMPT_TOKENS" \
        --max-tokens "$MAX_TOKENS"

    # Tear down.
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    sleep 2
}

run_one layer_first 32351
run_one page_first_kv_split 32352

echo ""
echo "bench complete."
exit 0