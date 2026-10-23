#!/usr/bin/env bash
# Compare GLM-5.2-w4a8c8 L2-replay TTFT between layer_first and
# page_first_kv_split. Run inside sglang-lzx-B103 with 16 free NPUs.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/GLM-5.2-w4a8c8}"
HOST="${HOST:-127.0.0.1}"
NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
LOG_DIR="${LOG_DIR:-/home/l00889655/sglang/bench_glm52_layout_ttft_16npu_logs}"
NUM_PROMPTS="${NUM_PROMPTS:-2}"
PROMPT_TOKENS="${PROMPT_TOKENS:-3000}"
MAX_TOKENS="${MAX_TOKENS:-16}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}"
MEASURE_ROUNDS="${MEASURE_ROUNDS:-5}"
L1_TOKENS="${L1_TOKENS:-8192}"
L2_RATIO="${L2_RATIO:-2.5}"
RUN_ORDER="${RUN_ORDER:-layer_first_first}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$LOG_DIR"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
export HCCL_BUFFSIZE=1024
export PYTHONPATH="/home/l00889655/sglang/sglang/python:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages:${PYTHONPATH:-}"
python -c "import acl; import sglang.srt.models.glm4_moe"

active_pid=""
stop_server() {
    if [ -n "$active_pid" ]; then
        kill "$active_pid" 2>/dev/null || true
        wait "$active_pid" 2>/dev/null || true
        active_pid=""
        sleep 2
    fi
}
trap stop_server EXIT

run_one() {
    local layout="$1"
    local port="$2"
    local server_log="$LOG_DIR/${layout}_server.log"
    local bench_log="$LOG_DIR/${layout}_bench.log"
    local result_json="$LOG_DIR/${layout}.json"
    local health_url="http://$HOST:$port/health"
    local ready=0

    echo "===== layout=$layout port=$port ====="
    echo "server log: $server_log"
    echo "bench log:  $bench_log"
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host "$HOST" \
        --port "$port" \
        --tp-size 16 \
        --ep 16 \
        --moe-a2a-backend deepep \
        --deepep-mode normal \
        --deepep-dispatcher-output-dtype bf16 \
        --enable-hicache \
        --hicache-mem-layout "$layout" \
        --hicache-io-backend kernel_ascend \
        --hicache-write-policy write_through \
        --hicache-ratio "$L2_RATIO" \
        --mem-fraction-static 0.85 \
        --max-total-tokens "$L1_TOKENS" \
        --device npu \
        --attention-backend ascend \
        --disable-cuda-graph \
        --log-level info \
        > "$server_log" 2>&1 &
    active_pid=$!

    for i in $(seq 1 360); do
        if curl -fsS "$health_url" >/dev/null 2>&1; then
            ready=1
            echo "server healthy after ~${i}s"
            break
        fi
        if ! kill -0 "$active_pid" 2>/dev/null; then
            echo "FATAL: server exited early. See $server_log"
            tail -80 "$server_log"
            return 1
        fi
        sleep 1
    done
    if [ "$ready" -ne 1 ]; then
        echo "FATAL: server did not become healthy. See $server_log"
        tail -120 "$server_log"
        return 1
    fi

    python "$SCRIPT_DIR/bench_glm52_layout_ttft_16npu.py" \
        --base-url "http://$HOST:$port" \
        --layout "$layout" \
        --server-log "$server_log" \
        --model-path "$MODEL_PATH" \
        --output-json "$result_json" \
        --num-prompts "$NUM_PROMPTS" \
        --prompt-tokens "$PROMPT_TOKENS" \
        --max-tokens "$MAX_TOKENS" \
        --warmup-rounds "$WARMUP_ROUNDS" \
        --measure-rounds "$MEASURE_ROUNDS" \
        --l1-tokens "$L1_TOKENS" \
        --l2-ratio "$L2_RATIO" \
        2>&1 | tee "$bench_log"

    stop_server
}

case "$RUN_ORDER" in
    layer_first_first)
        run_one layer_first 32351
        run_one page_first_kv_split 32352
        ;;
    page_first_first)
        run_one page_first_kv_split 32352
        run_one layer_first 32351
        ;;
    *)
        echo "Invalid RUN_ORDER: $RUN_ORDER" >&2
        exit 2
        ;;
esac

python "$SCRIPT_DIR/bench_glm52_layout_ttft_16npu.py" \
    --compare \
    --layer-json "$LOG_DIR/layer_first.json" \
    --page-json "$LOG_DIR/page_first_kv_split.json" \
    2>&1 | tee "$LOG_DIR/comparison.log"
