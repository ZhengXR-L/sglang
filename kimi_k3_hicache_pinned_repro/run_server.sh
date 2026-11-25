#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
if [[ "$MODEL_PATH" == "$PRIVATE_MODEL" ]]; then
  python3 -u "$HERE/prepare_model.py"
fi

QUANT_ARGS=()
if [[ "$QUANTIZATION" != "none" ]]; then
  QUANT_ARGS=(--quantization "$QUANTIZATION")
fi

sglang serve \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --device npu --attention-backend ascend --dtype bfloat16 \
  "${QUANT_ARGS[@]}" --tp-size "$TP_SIZE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
  --enable-hierarchical-cache --hicache-io-backend kernel_ascend \
  --hicache-mem-layout "$HICACHE_MEM_LAYOUT" \
  --hicache-write-policy write_through --hicache-ratio "$HICACHE_RATIO" \
  "${EXTRA_SERVER_ARGS[@]}" &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true' TERM INT
wait "$SERVER_PID"
