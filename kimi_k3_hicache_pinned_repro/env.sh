#!/usr/bin/env bash
# 唯一配置入口。按容器和 checkpoint 修改带「需修改」的默认值。
export SOURCE_MODEL="${SOURCE_MODEL:-/mnt/paas/weights/Kimi-K3-w4a8}" # 需修改：公共权重路径
export PRIVATE_MODEL="${PRIVATE_MODEL:-/home/l00889655/kimi_truncation}" # 需修改：私有目录
export SGLANG_ROOT="${SGLANG_ROOT:-$(cd -- "$HERE/.." && pwd)}" # 需修改
export ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/latest/set_env.sh}" # 需修改
export NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7}" # 需修改
export MODEL_PATH="${MODEL_PATH:-$PRIVATE_MODEL}"
export PORT="${PORT:-30000}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
export OBSERVE_SECONDS="${OBSERVE_SECONDS:-10}"
export PREFIX_TOKENS="${PREFIX_TOKENS:-2048}"
export REQUEST_COUNT="${REQUEST_COUNT:-4}"
export QUANTIZATION="${QUANTIZATION:-modelslim}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
export MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-32}"
export HICACHE_RATIO="${HICACHE_RATIO:-2.0}"
export HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_kv_split}"
export SGLANG_HICACHE_HOST_MEM_BACKEND="${SGLANG_HICACHE_HOST_MEM_BACKEND:-}"
if [[ -z "$SGLANG_HICACHE_HOST_MEM_BACKEND" ]]; then
  unset SGLANG_HICACHE_HOST_MEM_BACKEND
fi
export SGLANG_NPU_HICACHE_MAMBA_IO="${SGLANG_NPU_HICACHE_MAMBA_IO:-sync}"

# 保留原故障配置需要的并行、MoE 等选项。每个参数独立引用。
EXTRA_SERVER_ARGS=(
  # --attn-dp-size 4 --enable-dp-lm-head
  # --moe-a2a-backend deepep --deepep-mode auto
)

set +u
source "$ASCEND_ENV" || return
set -u
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export PYTHONPATH="$HERE/trace:$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
IFS=, read -r -a devices <<< "$NPU_DEVICES"
export TP_SIZE="${#devices[@]}"
