export PYTHONPATH="/home/l00889655/sglang/sglang/python${PYTHONPATH:+:$PYTHONPATH}"

python3 verify_hicache_retry_interval_npu.py \
  --interval 8 \
  --port 7777 \
  --log /tmp/hicache-retry-i8.log 