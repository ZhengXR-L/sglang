#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"

if [[ "$USE_IMAGE_CODE" == 1 ]]; then
  cd /workspace
  python3 - <<'PYTHON'
from importlib.util import find_spec
from pathlib import Path
from sysconfig import get_path

for module, root in (
    ("sglang", Path("/sgl-workspace/sglang/python/sglang")),
    ("sgl_kernel_npu", Path(get_path("purelib")) / "sgl_kernel_npu"),
):
    spec = find_spec(module)
    if not spec or not spec.origin or not Path(spec.origin).resolve().is_relative_to(root):
        raise SystemExit(f"{module} did not resolve to the image package: {spec}")
    print(f"{module}: {spec.origin}", flush=True)
PYTHON
fi

IMAGE_ARGS=()
if [[ "$USE_IMAGE_CODE" == 1 ]]; then
  IMAGE_ARGS=(--page-size "$PAGE_SIZE" --hicache-mem-layout page_first_direct
              --hicache-write-policy write_through
              --disable-decode-cuda-graph --disable-prefill-cuda-graph)
fi

sglang serve \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --device npu \
  --tp-size "$TP_SIZE" \
  --enable-hierarchical-cache \
  --enable-metrics \
  --hicache-ratio "$HICACHE_RATIO" \
  --hicache-io-backend kernel_ascend \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --mamba-ssm-dtype bfloat16 \
  --context-length "$CONTEXT_LENGTH" "${IMAGE_ARGS[@]}" &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true' TERM INT
wait "$SERVER_PID"
