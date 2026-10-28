#!/usr/bin/env bash
set -euo pipefail
BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export LOG_DIR="${LOG_DIR:-$BASE/target_verify_default}"
test ! -e "$LOG_DIR"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/run.log" 2>&1
export SGL_SRC_HOME=/home/l00889655/sglang/sglang
export MODEL_PATH=/mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8
export PORT=${PORT:-32367}
export BASE_URL="http://127.0.0.1:$PORT"
export NPU_DEVICES=4,5,6,7,8,9,10,11
export TP_SIZE=8
export DP_RANKS=8
export EP_SIZE=8
export QUANTIZATION=modelslim
export KV_CACHE_DTYPE=bfloat16
export MAX_TOTAL_TOKENS=66000
export CHUNKED_PREFILL_SIZE=32768
export MEM_FRACTION_STATIC=0.8
export HICACHE_RATIO=2.0
export HICACHE_LAYOUT="${HICACHE_LAYOUT:-layer_first}"
export SGLANG_DSV4_AIV_TRACE=1
export ENABLE_MTP=0
export DISABLE_GRAPHS=1
export RANDOM_SEED=0
export MOE_A2A_BACKEND=deepep
export SGLANG_ENABLE_SPEC_V2=0
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_NPU_USE_MULTI_STREAM=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEPEP_HCCL_BUFFSIZE=1024
export SGLANG_ZBAL_LOCAL_MEM_SIZE=0
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=1
export RESULT_DIR="$LOG_DIR/results"
export SERVER_PID_FILE="$LOG_DIR/server.pid"
export READY_TIMEOUT=900

cleanup() {
    status=$?
    trap - EXIT
    if [[ -f "$SERVER_PID_FILE" ]]; then
        pid=$(cat "$SERVER_PID_FILE")
        if [[ -r "/proc/$pid/cmdline" ]]; then
            command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
            case "$command_line" in
                *sglang.launch_server*"--port $PORT"*) kill "$pid" 2>/dev/null || true ;;
                *) echo "Not stopping unexpected PID $pid: $command_line" ;;
            esac
        fi
        for unused in $(seq 1 30); do
            if ! kill -0 "$pid" 2>/dev/null; then break; fi
            if [[ -r "/proc/$pid/stat" ]] && [[ "$(awk '{print $3}' "/proc/$pid/stat")" == Z ]]; then break; fi
            sleep 1
        done
    fi
    printf '%s\n' "$status" > "$LOG_DIR/exit_code"
    exit "$status"
}
trap cleanup EXIT
test "$(git -C "$SGL_SRC_HOME" branch --show-current)" = layer_first_ascendc_sparse_copy
bash "$BASE/test_hicache/test_cached_token.sh" 65536 8 100 16