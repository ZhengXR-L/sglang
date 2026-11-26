#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?usage: bash run_test.sh <output_dir>}
mkdir -p -- "$OUT"
OUT=$(cd -- "$OUT" && pwd)
: > "$OUT/server.log"
: > "$OUT/eval.log"
: > "$OUT/report.log"

RESULT="准备失败"
SERVER_GROUP=
finish() {
  local rc=$?
  if [[ -n "$SERVER_GROUP" ]]; then
    kill -TERM -- "-$SERVER_GROUP" 2>/dev/null || true
    for ((i = 0; i < 5; i++)); do
      kill -0 -- "-$SERVER_GROUP" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_GROUP" 2>/dev/null || true
    wait "$SERVER_GROUP" 2>/dev/null || true
  fi
  if ! python3 "$HERE/report.py" "$OUT" "$rc" "$RESULT" > "$OUT/report.log" 2>&1; then
    printf '# Kimi K3 HiCache L2 复现\n\n%s（退出码 %s）\n' "$RESULT" "$rc" > "$OUT/report.md"
    if (( rc == 0 )); then rc=2; fi
  fi
  printf '%s\n输出目录 %s\n' "$RESULT" "$OUT"
  printf '  %s\n' "$OUT/server.log" "$OUT/eval.log" "$OUT/report.log" "$OUT/report.md"
  exit "$rc"
}
trap finish EXIT
trap 'RESULT="测试被中断"; exit 130' INT TERM
if ! source "$HERE/env.sh" >> "$OUT/eval.log" 2>&1; then
  RESULT="环境配置失败，见 eval.log"
  exit 2
fi

BASE_URL="http://127.0.0.1:$PORT"
if curl -s --max-time 2 "$BASE_URL/health" >/dev/null; then
  RESULT="目标端口已有 HTTP 服务"
  exit 2
fi

target_error() {
  grep -Eq 'RuntimeError: host buffer must use pinned memory[[:space:]]*$' "$OUT/server.log" &&
    grep -Fq 'in transfer_mamba_state' "$OUT/server.log" &&
    grep -Fq 'in backup_from_device_all_layer' "$OUT/server.log"
}

printf '等待 SGLang 启动\n'
setsid bash "$HERE/run_server.sh" > "$OUT/server.log" 2>&1 &
SERVER_GROUP=$!
deadline=$((SECONDS + STARTUP_TIMEOUT))
ready=0
while (( SECONDS < deadline )); do
  kill -0 "$SERVER_GROUP" 2>/dev/null || break
  if curl -fsS --max-time 2 "$BASE_URL/health" >> "$OUT/eval.log" 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

request_rc=0
if (( ready )); then
  printf '发送共享前缀请求并检查写回日志\n'
  python3 -u "$HERE/run_case.py" "$BASE_URL" >> "$OUT/eval.log" 2>&1 || request_rc=$?
  deadline=$((SECONDS + OBSERVE_SECONDS))
  while (( SECONDS < deadline )); do
    target_error && break
    sleep 1
  done
fi

if target_error; then
  RESULT="已复现 transfer_mamba_state 的 pinned memory 异常"
  exit 0
fi
if (( ! ready )); then
  RESULT="服务启动失败或超时，未观察到目标异常"
  exit 2
fi
if (( request_rc )) || ! kill -0 "$SERVER_GROUP" 2>/dev/null; then
  RESULT="请求失败或服务退出，未观察到目标异常"
  exit 2
fi
if grep -q MAMBA_OP_OK "$OUT/server.log"; then
  RESULT="请求完成，已确认 Mamba 写回算子执行成功"
  exit 0
fi
RESULT="请求完成，未观察到目标异常或算子成功记录"
exit 1
