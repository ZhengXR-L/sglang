#!/usr/bin/env bash
# =============================================================================
# 环境准备：安装 Rust 工具链/protoc、系统编译依赖、Python gRPC 客户端依赖
#
# 用法：
#   ./prepare_env.sh
#
# 可用环境变量覆盖：
#   SGLANG_REPO_DIR    sglang 仓库根目录（默认 /home/l00889655/sglang/sglang_work2/sglang）
#
# 前置条件：apt 可用（root 或有 sudo）
# =============================================================================

set -euo pipefail

SGLANG_REPO_DIR="${SGLANG_REPO_DIR:-/home/l00889655/sglang/sglang_work2/sglang}"

echo "===== 环境准备：安装 Rust 依赖 ====="

# 系统编译依赖：C 编译器(make/gcc)、pkg-config、Python 头文件、curl
if command -v apt-get >/dev/null 2>&1; then
    if [ "$(id -u)" = "0" ]; then SUDO=""; else SUDO="sudo"; fi
    ${SUDO} apt-get update -y
    ${SUDO} apt-get install -y --no-install-recommends \
        build-essential pkg-config python3-dev curl ca-certificates
else
    echo "警告: 未找到 apt-get，请手动安装 build-essential/python3-dev 等系统依赖"
fi

# Rust 工具链 + protoc（仓库自带脚本，幂等；rustup 装到用户目录，protoc 需 root）
bash "$SGLANG_REPO_DIR/scripts/ci/utils/install_rust_protoc.sh"

# Python gRPC 客户端依赖（grpcio 提供 import grpc，grpcio-tools 提供 grpc_tools 生成 stub）
python3 -m pip install --upgrade grpcio grpcio-tools

echo "===== 环境准备完成 ====="