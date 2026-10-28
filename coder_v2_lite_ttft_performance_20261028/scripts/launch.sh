#!/usr/bin/env bash
set -u
SCRIPT=/home/l00889655/sglang/sglang/coder_v2_lite_ttft_performance/scripts/run_compare.py
BASE=${1:-/home/l00889655/sglang/coder_v2_lite_ttft_$(date +%Y%m%d_%H%M%S)}
python3 -u "$SCRIPT" "$BASE" 10 > "${BASE}_controller.log" 2>&1
echo $? > "${BASE}_exit_code"
