#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REQ_LEN=${1:-65536}
N=${2:-8}
HIT=${3:-100}
POP=${4:-$((2*N))}
export LOG_DIR=${LOG_DIR:-/home/cx/log/dsv4_hicache_$(date +%Y%m%d_%H%M%S)}
export SERVER_LOG="$LOG_DIR/server.log"
export SERVER_PID_FILE="$LOG_DIR/server.pid"

bash "$SCRIPT_DIR/run_l2_server.sh"

bash "$SCRIPT_DIR/test_l2.sh" "$REQ_LEN" "$N" "$HIT" "$POP"
