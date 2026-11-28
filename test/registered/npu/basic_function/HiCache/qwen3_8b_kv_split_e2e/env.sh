#!/usr/bin/env bash
# 唯一配置入口；跨机器使用时修改标为「需修改」的路径。

# B060 mode uses the container's toolchain and installed kernel package.
export USE_IMAGE_CODE="${USE_IMAGE_CODE:-0}"
if [[ "$USE_IMAGE_CODE" == 1 ]]; then
  export SGLANG_ROOT=/sgl-workspace/sglang
  export MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/Qwen3-8B}"
  export NPU_DEVICES="${NPU_DEVICES:-12,13}"
  export PORT="${PORT:-47312}"
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
  export SGLANG_HICACHE_HOST_MEM_BACKEND="${SGLANG_HICACHE_HOST_MEM_BACKEND:-acl}"
fi
export MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B}" # 需修改
export SGLANG_ROOT="${SGLANG_ROOT:-/path/to/sglang}" # 需修改
export KERNEL_ROOT="${KERNEL_ROOT:-/path/to/sgl-kernel-npu}" # 需修改
export KERNEL_PYTHON_PATH="${KERNEL_PYTHON_PATH:-$KERNEL_ROOT/python/sgl_kernel_npu}" # A5 指向已安装 wheel 的 site-packages
export ASCEND_ENV_SCRIPT="${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/latest/set_env.sh}" # 需修改
export NPU_DEVICES="${NPU_DEVICES:-0}" # 需修改：空闲物理卡，逗号分隔
export PORT="${PORT:-30000}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
export PAGE_SIZE="${PAGE_SIZE:-128}"
export PROMPT_TOKENS="${PROMPT_TOKENS:-1024}"
export OUTPUT_TOKENS="${OUTPUT_TOKENS:-8}"
export FILLER_REQUESTS="${FILLER_REQUESTS:-8}"
export HICACHE_RATIO="${HICACHE_RATIO:-4}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
export SGLANG_HICACHE_HOST_MEM_BACKEND="${SGLANG_HICACHE_HOST_MEM_BACKEND:-acl}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True,pin_memory_expandable_segments:True}"

if [[ "$USE_IMAGE_CODE" != 1 ]]; then
  set +u
  source "$ASCEND_ENV_SCRIPT"
  set -u
fi
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
IFS=, read -r -a _npu_cards <<< "$NPU_DEVICES"
export TP_SIZE="${#_npu_cards[@]}"
if [[ "$USE_IMAGE_CODE" != 1 ]]; then
  export PYTHONPATH="$SGLANG_ROOT/python:$KERNEL_PYTHON_PATH${PYTHONPATH:+:$PYTHONPATH}"
fi
