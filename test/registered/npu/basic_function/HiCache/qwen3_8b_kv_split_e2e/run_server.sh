#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
LAYOUT=${1:?usage: bash run_server.sh page_first_direct}

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

GRAPH_ARGS=(--disable-cuda-graph)
if [[ "$USE_IMAGE_CODE" == 1 ]]; then
  GRAPH_ARGS=(--disable-decode-cuda-graph --disable-prefill-cuda-graph)
fi

sglang serve \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --device npu --attention-backend ascend \
  --tp-size "$TP_SIZE" --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" --page-size "$PAGE_SIZE" \
  --enable-hierarchical-cache --hicache-ratio "$HICACHE_RATIO" \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel_ascend --hicache-mem-layout "$LAYOUT" \
  --enable-metrics "${GRAPH_ARGS[@]}" &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true' TERM INT
wait "$SERVER_PID"
