#!/usr/bin/env bash
# =============================================================================
# 验证 --grpc-response-timeout-secs：Rust 单元测试 + Qwen3-0.6B 端到端
#
# 用法：
#   ./run_verification.sh            # 依次执行 unit + e2e
#   ./run_verification.sh unit       # 只跑 Rust 单元测试
#   ./run_verification.sh e2e        # 只跑 Qwen3-0.6B 端到端
#
# 可用环境变量覆盖（A5 环境路径不同时务必覆盖）：
#   SGLANG_REPO_DIR    sglang 仓库根目录（默认 /home/l00889655/sglang/sglang_work2/sglang）
#   MODEL_PATH         模型权重路径（默认 /home/weights/Qwen3-0.6B）
#   DEVICE            --device（默认 npu）
#   HOST              服务绑定地址（默认 127.0.0.1）
#   HTTP_PORT         服务 HTTP 端口（默认 30000）
#   GRPC_PORT         服务 gRPC 端口（默认 50052）
#   GRPC_TIMEOUT_SECS 待验证的超时值（默认 42，非默认值，用于确认参数被读取）
#
# 前置条件：
#   - 已执行 prepare_env.sh（或已装好 Rust 工具链 + grpcio/grpcio-tools）
#   - sglang 已安装（可 import sglang 且能 python3 -m sglang.launch_server）
# =============================================================================

set -euo pipefail

SGLANG_REPO_DIR="${SGLANG_REPO_DIR:-/home/l00889655/sglang/sglang_work2/sglang}"
MODEL_PATH="${MODEL_PATH:-/home/weights/Qwen3-0.6B}"
DEVICE="${DEVICE:-npu}"
HOST="${HOST:-127.0.0.1}"
HTTP_PORT="${HTTP_PORT:-30000}"
GRPC_PORT="${GRPC_PORT:-50052}"
GRPC_TIMEOUT_SECS="${GRPC_TIMEOUT_SECS:-42}"

SERVER_PID=""

# -----------------------------------------------------------------------------
# Rust 单元测试（超时逻辑）
# -----------------------------------------------------------------------------
run_unit_tests() {
    echo "===== 运行 Rust 单元测试 ====="
    cd "$SGLANG_REPO_DIR/rust"
    # 跑 sglang-grpc crate 的全部单测（含 recv_chunk_with_timeout 两个超时用例）。
    # 如需只跑超时用例：cargo test -p sglang-grpc recv_chunk
    cargo test -p sglang-grpc
    echo "===== 单元测试通过 ====="
}

# -----------------------------------------------------------------------------
# Qwen3-0.6B 端到端验证
# -----------------------------------------------------------------------------
cleanup_server() {
    [ -n "$SERVER_PID" ] || return 0
    echo "清理服务 (PID=$SERVER_PID)..."
    # 用 sglang 自带的进程树清理，失败则兜底 kill -9
    python3 -c "from sglang.srt.utils import kill_process_tree; kill_process_tree($SERVER_PID)" 2>/dev/null \
        || kill -9 "$SERVER_PID" 2>/dev/null \
        || true
}

run_e2e() {
    echo "===== Qwen3-0.6B 端到端验证 ====="
    echo "模型: $MODEL_PATH"
    echo "设备: $DEVICE   gRPC: $HOST:$GRPC_PORT   HTTP: $HOST:$HTTP_PORT"
    echo "待验证 --grpc-response-timeout-secs = $GRPC_TIMEOUT_SECS"

    # 前置检查：sglang 可用
    python3 -c "import sglang" || { echo "错误: sglang 未安装，请先安装 sglang"; exit 1; }

    # 后台启动服务（原生 gRPC 由 --grpc-port 启用）
    python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host "$HOST" \
        --port "$HTTP_PORT" \
        --grpc-port "$GRPC_PORT" \
        --grpc-response-timeout-secs "$GRPC_TIMEOUT_SECS" \
        --device "$DEVICE" \
        --trust-remote-code \
        >/tmp/sglang_grpc_e2e_server.log 2>&1 &
    SERVER_PID=$!
    trap cleanup_server EXIT
    echo "服务已后台启动 (PID=$SERVER_PID)，日志: /tmp/sglang_grpc_e2e_server.log"

    # 用内嵌 Python 生成 gRPC stub 并验证（读取环境变量）
    SGLANG_REPO_DIR="$SGLANG_REPO_DIR" \
    HOST="$HOST" \
    GRPC_PORT="$GRPC_PORT" \
    GRPC_TIMEOUT_SECS="$GRPC_TIMEOUT_SECS" \
    python3 - <<'PY'
import json
import os
import sys
import tempfile
import time

import grpc

repo_dir = os.environ["SGLANG_REPO_DIR"]
host = os.environ["HOST"]
grpc_port = int(os.environ["GRPC_PORT"])
expect_timeout = int(os.environ["GRPC_TIMEOUT_SECS"])

# 1) 从 sglang.proto 动态生成 SglangService stub
out_dir = tempfile.mkdtemp(prefix="sglang_grpc_stub_")
proto_dir = os.path.join(repo_dir, "proto")
proto_file = os.path.join(proto_dir, "sglang", "runtime", "v1", "sglang.proto")

import grpc_tools
from grpc_tools import protoc
# grpc_tools 内置的 well-known protos（含 google/protobuf/struct.proto）
grpc_tools_include = os.path.join(os.path.dirname(grpc_tools.__file__), "_proto")
try:
    protoc.main([
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"-I{grpc_tools_include}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        proto_file,
    ])
except SystemExit:
    pass

# 补齐包结构 __init__.py，使 sglang.runtime.v1 可被绝对导入
for sub in ("sglang", "sglang/runtime", "sglang/runtime/v1"):
    init_file = os.path.join(out_dir, *sub.split("/"), "__init__.py")
    with open(init_file, "w"):
        pass

sys.path.insert(0, out_dir)
from sglang.runtime.v1 import sglang_pb2 as pb2
from sglang.runtime.v1 import sglang_pb2_grpc as pb2_grpc

channel = grpc.insecure_channel(f"{host}:{grpc_port}")
stub = pb2_grpc.SglangServiceStub(channel)

# 2) 轮询等待 gRPC 服务就绪（模型加载完成）
ready = False
for _ in range(150):  # 最多约 5 分钟
    try:
        hc = stub.HealthCheck(pb2.HealthCheckRequest())
        if getattr(hc, "healthy", False):
            ready = True
            break
    except grpc.RpcError:
        pass
    time.sleep(2)
if not ready:
    raise SystemExit("gRPC 服务未在预期时间内就绪")

print("[OK] HealthCheck healthy=True")

# 3) 验证 --grpc-response-timeout-secs 端到端传递
info = json.loads(stub.GetServerInfo(pb2.GetServerInfoRequest()).json_info)
got = info.get("grpc_response_timeout_secs")
print(f"[OK] GetServerInfo grpc_response_timeout_secs = {got} (期望 {expect_timeout})")
assert got == expect_timeout, f"grpc_response_timeout_secs 不匹配: {got} != {expect_timeout}"

# 4) 验证 TextGenerate 正常收发
req = pb2.TextGenerateRequest(
    text="What is the capital of France?",
    sampling_params=pb2.SamplingParams(max_new_tokens=16),
)
text = "".join(chunk.text for chunk in stub.TextGenerate(req))
assert text.strip(), "TextGenerate 返回空文本"
print(f"[OK] TextGenerate 返回: {text.strip()!r}")

print("===== 端到端验证通过 =====")
PY
}

# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------
case "${1:-all}" in
    unit) run_unit_tests ;;
    e2e)  run_e2e ;;
    all)  run_unit_tests; run_e2e ;;
    *)
        echo "用法: $0 [unit|e2e|all]"
        exit 1
        ;;
esac