#!/usr/bin/env bash
set -u
SCRIPT=/home/l00889655/sglang/sglang/glm52_ttft_performance/scripts/run_compare.py
BASE=${1:-/home/l00889655/sglang/glm52_ttft_$(date +%Y%m%d_%H%M%S)}
python3 -u "$SCRIPT" "$BASE" "${2:-4,5,6,7,8,9,14,15}" > "${BASE}_controller.log" 2>&1
echo $? > "${BASE}_exit_code"
