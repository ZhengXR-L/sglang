export PYTHONPATH="/home/l00889655/sglang/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$(mktemp -d /tmp/hicache-retry.XXXXXX)"
export PYTHONUNBUFFERED=1

INTERVAL=8
set -o pipefail

sglang serve \
  --model-path /home/weights/Qwen3-0.6B \
  --device npu \
  --mem-fraction-static 0.6 \
  --attention-backend ascend \
  --host 127.0.0.1 \
  --port 7777 \
  --enable-hierarchical-cache \
  --hicache-storage-backend file \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-prefetch-retry-poll-interval "$INTERVAL" \
  --hicache-storage-prefetch-retry-max-attempts 1 \
  --max-running-requests 1 \
  --log-level debug \
  2>&1 | tee "/tmp/hicache-retry-i${INTERVAL}.log"