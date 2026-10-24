#!/bin/bash
# layer_first DeepSeek-Coder（通用 MLA/AscendC）roundtrip 的 profiling 采集入口。
# 在 sglang-lzx-B103 容器内运行。
#
# 作用：启动 DeepSeek-Coder layer_first + hicache（kernel_ascend + write_through）
# server 后，调用 profile_layer_first_roundtrip.py，分别采集 D2H/H2D 两个
# 方向、短/长两档的 profile trace（profile_by_stage + detailed_annotations）。
#
# 用法：
#   bash profile_layer_first_roundtrip.sh \
#       [--short-doc-tokens 2048] [--long-doc-tokens 10000] [--num-steps 8] ...
#
# 输出目录默认为 /home/l00889655/sglang/profiler/{short,long}/{d2h,h2d}/。

set -eo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/DeepSeek-Coder-V2-Lite-Instruct}"
PORT="${PORT:-32345}"
HOST="${HOST:-127.0.0.1}"
NPU_DEVICES="${NPU_DEVICES:-8}"
BASE_URL="http://$HOST:$PORT"
OUTPUT_DIR="${OUTPUT_DIR:-/home/l00889655/sglang/profiler}"
LOG_DIR="${LOG_DIR:-/home/l00889655/sglang/e2e_layer_first_logs}"
HICACHE_LAYOUT="${HICACHE_LAYOUT:-layer_first}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
SERVER_LOG="$LOG_DIR/profile_roundtrip_server.log"

export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric
# 容器默认 PYTHONPATH 不含 CANN site-packages，会导致 import acl 失败、deepseek_v2
# 原生实现回退 Transformers 后触发 KeyError: 'sglang' 崩溃。必须显式补上 CANN 路径。
export PYTHONPATH="/home/l00889655/sglang/sglang/python:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages:${PYTHONPATH:-}"

echo "=== profile: layer_first DeepSeek-Coder L1<->L2 roundtrip (D2H + H2D) ==="
echo "model:      $MODEL_PATH"
echo "devices:    $NPU_DEVICES  (tp=1)"
echo "base url:   $BASE_URL"
echo "output dir: $OUTPUT_DIR"
echo "server log: $SERVER_LOG"

# 1. 启动 server（host pool ratio 2.5 大于 device pool，保证 L2 能在 L1 淘汰后
#    仍保留 doc）。
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp 1 \
    --enable-hicache \
    --hicache-mem-layout "$HICACHE_LAYOUT" \
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

# 2. 等待 server 健康。
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
        echo "FATAL: server 进程提前退出。见 $SERVER_LOG"
        tail -80 "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done

if [ "$ready" -ne 1 ]; then
    echo "FATAL: server 未在限定时间内健康。见 $SERVER_LOG"
    tail -120 "$SERVER_LOG"
    exit 1
fi

# 3. 采集 profile（warmup + 短/长两档，各自 D2H/H2D 分方向）。
python "$(dirname "$0")/profile_layer_first_roundtrip.py" \
    --base-url "$BASE_URL" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$MODEL_PATH" \
    "$@"

echo "done."
exit 0