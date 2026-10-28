#!/bin/bash
set -euo pipefail
# usage: test.sh <请求全长 req_len> <measure N> [hit:100|50] [populate条数 POP]
#   L2 测试：N 条 A 完成 → POP-N 条不同前缀 B 完成 → 只 replay N 条 A。
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REQ_LEN=${1:?usage: test.sh <req_len> <N> [hit] [pop]}
N=${2:?}
HIT=${3:-100}
POP=${4:-$((2*N))}
if [ "$HIT" != "100" ] && [ "$HIT" != "50" ]; then echo "hit must be 100/50" >&2; exit 1; fi
P=$REQ_LEN
LEN_K=$((P/1024))

SGL_SRC_HOME=${SGL_SRC_HOME:-/home/cx/sglang}
MODEL_PATH=${MODEL_PATH:-/mnt/weight/DeepSeek-V4-Flash-DSpark}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT:-30000}}
DP_RANKS=${DP_RANKS:-8}
if (( POP < 2 * N )); then echo "POP must be >= 2*N for L2 eviction validation" >&2; exit 1; fi

echo "=== WAITING FOR SERVER... ==="
T1=10; MAX=${READY_TIMEOUT:-600}; URL=$BASE_URL/health; S=$SECONDS
while :; do
  if [[ -n "${SERVER_PID_FILE:-}" && -f "$SERVER_PID_FILE" ]] && ! kill -0 "$(cat "$SERVER_PID_FILE")" 2>/dev/null; then
    echo "[ERR] launched server exited; see ${SERVER_LOG:-server.log}" >&2; exit 1
  fi
  code=$(curl --max-time 5 -s -o /dev/null -w "%{http_code}" "$URL" 2>/dev/null || true)
  if [ "$code" = "200" ]; then echo "[OK] ready after $((SECONDS-S))s"; break; fi
  if [ $((SECONDS-S)) -ge "$MAX" ]; then echo "[TIMEOUT] not ready within ${MAX}s" >&2; exit 1; fi
  sleep "$T1"
done

curl -sf -X POST "$BASE_URL/flush_cache" -m 10 || { echo "[ERR] flush failed" >&2; exit 1; }

LOG_PATH=${SERVER_LOG:-${LOG_DIR:+$LOG_DIR/server.log}}
if [[ -z "$LOG_PATH" ]]; then echo "SERVER_LOG or LOG_DIR is required" >&2; exit 1; fi
echo "[INFO] server-log: $LOG_PATH"
TAG="L2_${REQ_LEN}_${HIT}"

RESULT_DIR=${RESULT_DIR:-$(dirname "$LOG_PATH")/${TAG}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RESULT_DIR"

echo "=== [L2] populate A=$N B=$((POP-N)); replay A=$N (prefix=$P, hit=$HIT) ==="
PYTHONPATH="$SGL_SRC_HOME/python:${PYTHONPATH:-}" python3 -u "$SCRIPT_DIR/bench_ids_dsv4.py" \
    --model-path "$MODEL_PATH" --base-url "$BASE_URL" \
    --input-len "$P" --num-prompts "$N" --populate-prompts "$POP" --output-len 8 \
    --route roundrobin --dp-ranks $DP_RANKS --concurrency $DP_RANKS --pop-conc $DP_RANKS --hit $HIT \
    --skip-measure --require-l2-hit --output-dir "$RESULT_DIR" --tag "${TAG}" \
    --server-log "$LOG_PATH" --seed-base 60000

echo "=== [L2] DONE ==="
