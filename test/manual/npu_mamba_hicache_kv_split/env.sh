#!/usr/bin/env bash
# 唯一配置入口。按实际环境修改标记为「需修改」的默认值。

export SGLANG_ROOT="${SGLANG_ROOT:-/home/l00889655/sglang/sglang}" # 需修改
# export KERNEL_ROOT="${KERNEL_ROOT:-/home/l00889655/sglang/sgl-kernel-npu}" # 需修改
export MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/Qwen3.5-4B}" # 需修改

export NPU_DEVICES="${NPU_DEVICES:-9}"  # 空闲物理卡 ID 列表
export TP_SIZE="$(awk -F, '{print NF}' <<< "$NPU_DEVICES")"
export PORT="${PORT:-30000}"

# 驱逐压力参数：target 请求驱逐后重放，filler 请求占满 device mamba slot
export TARGET_PROMPT_CHARS="${TARGET_PROMPT_CHARS:-20000}"
export TARGET_NEW_TOKENS="${TARGET_NEW_TOKENS:-64}"
export NUM_FILLER="${NUM_FILLER:-5}"
export FILLER_PROMPT_CHARS="${FILLER_PROMPT_CHARS:-12000}"
export FILLER_NEW_TOKENS="${FILLER_NEW_TOKENS:-8}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"

# 服务端容量控制：小 device 池 + 大 host 池，制造 L2 驱逐与回读
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export HICACHE_RATIO="${HICACHE_RATIO:-4}"
export GSM8K_HICACHE_RATIO="${GSM8K_HICACHE_RATIO:-8}"

# GSM8K 精度与逐题 Host 回迁用例
export GSM8K_TRAIN_PATH="${GSM8K_TRAIN_PATH:-/home/l00889655/sglang/gsm8k_train.jsonl}" # 需填写 train.jsonl
export GSM8K_TEST_PATH="${GSM8K_TEST_PATH:-/home/l00889655/sglang/gsm8k_test.jsonl}" # 需填写 test.jsonl
export GSM8K_NUM_QUESTIONS="${GSM8K_NUM_QUESTIONS:-200}"
export GSM8K_NUM_SHOTS="${GSM8K_NUM_SHOTS:-8}"
export GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-512}"
export GSM8K_ACCURACY="${GSM8K_ACCURACY:-}"
export TEST_CASE="${TEST_CASE:-gsm8k}"
export UNIQUE_PAD_TOKENS="${UNIQUE_PAD_TOKENS:-640}"
export PAGE_SIZE="${PAGE_SIZE:-128}"
export FILLER_PROMPT_TOKENS="${FILLER_PROMPT_TOKENS:-4096}"
export MAX_FILLER_REQUESTS="${MAX_FILLER_REQUESTS:-12}"
export GLOBAL_FILLER_CAP="${GLOBAL_FILLER_CAP:-0}"

# Ascend 运行环境；set_env.sh 会引用未定义变量，source 期间临时关闭 -u。
set +u
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"

export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
# 使用包含 state direct 传输及模型算子的构建包；安装 wheel 时可覆盖。
export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
