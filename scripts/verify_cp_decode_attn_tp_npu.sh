#!/usr/bin/env bash
# Launch DeepSeek V4 on Ascend with CP decode attention TP enabled.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local DeepSeek V4 model directory}"

tp_size="${TP_SIZE:-8}"
port="${PORT:-7777}"

# The DeepSeek V4 Flash NPU W8A8 recipe uses BF16 for the wo_a absorb GEMM.
# The FP8 path expects weight_scale_inv, which ModelSlim W8A8 does not create.
export SGLANG_OPT_FP8_WO_A_GEMM=0

exec sglang serve \
  --model-path "$MODEL_PATH" \
  --trust-remote-code \
  --device npu \
  --attention-backend dsv4 \
  --tp-size "$tp_size" \
  --dp-size 1 \
  --attn-cp-size "$tp_size" \
  --enable-prefill-cp \
  --cp-strategy zigzag \
  --disable-cuda-graph \
  --host 127.0.0.1 \
  --port "$port" \
  --enable-cp-decode-attn-tp \
  "$@"
