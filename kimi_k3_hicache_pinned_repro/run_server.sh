#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
if [[ "$MODEL_PATH" == "$PRIVATE_MODEL" ]]; then
  python3 -u "$HERE/prepare_model.py"
fi

# 使用已裁剪 checkpoint 时也检查层数和类型覆盖。
python3 - "$MODEL_PATH" "$PREFIX_TOKENS" "$CONTEXT_LENGTH" <<'CHECK'
import json
import sys
from pathlib import Path

config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
text = config.get("text_config", config)
if text["num_hidden_layers"] != 5:
    raise SystemExit("本用例要求五层 checkpoint，请先准备私有减层目录")
attention = text["linear_attn_config"]
for name in ("kda_layers", "full_attn_layers"):
    kept = [layer for layer in attention[name] if 1 <= layer <= 5]
    if not kept:
        raise SystemExit(f"前五层没有覆盖 {name}")
    print(f"{name}={kept}", flush=True)
if int(sys.argv[2]) + 1 + 8 > int(sys.argv[3]):
    raise SystemExit("CONTEXT_LENGTH 至少需要覆盖 PREFIX_TOKENS + 1 + 8")
CHECK

QUANT_ARGS=()
if [[ "$QUANTIZATION" != "none" ]]; then
  QUANT_ARGS=(--quantization "$QUANTIZATION")
fi

sglang serve \
  --model-path "$MODEL_PATH" --tokenizer-path "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --device npu --attention-backend ascend --dtype bfloat16 \
  "${QUANT_ARGS[@]}" --tp-size "$TP_SIZE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
  --enable-hierarchical-cache --enable-cache-report --hicache-io-backend kernel_ascend \
  --hicache-mem-layout "$HICACHE_MEM_LAYOUT" \
  --hicache-write-policy write_through --hicache-ratio "$HICACHE_RATIO" \
  --disable-decode-cuda-graph --disable-prefill-cuda-graph --context-length "$CONTEXT_LENGTH" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" --max-running-requests "$MAX_RUNNING_REQUESTS" \
  "${EXTRA_SERVER_ARGS[@]}" &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true' TERM INT
wait "$SERVER_PID"
