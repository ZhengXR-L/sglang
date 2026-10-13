#!/usr/bin/env bash
# Run the BF16 QSA E2E test once all eight local Ascend NPUs are idle.

set -u -o pipefail

container="${QSA_TEST_CONTAINER:-test-fby}"
test_file="${QSA_TEST_FILE:-/tmp/codex_qsa_bf16_20261013.py}"
pythonpath="${QSA_TEST_PYTHONPATH:-/sgl-workspace/sglang/python}"
prompt_repeat="${QSA_TEST_PROMPT_REPEAT:-320}"
log_dir="${QSA_TEST_LOG_DIR:-/tmp/codex_qsa_bf16_20261013_run}"
poll_seconds="${QSA_TEST_POLL_SECONDS:-5}"
mkdir -p "$log_dir"

exec 9>"$log_dir/lock"
if ! flock -n 9; then
    echo "A QSA test waiter is already running for $log_dir" >&2
    exit 1
fi

status_file="$log_dir/status"
printf 'waiting\n' >"$status_file"
printf '%s Waiting for all 8 NPUs to become idle\n' "$(date -Is)" >>"$log_dir/runner.log"

idle_count() {
    local info
    if ! info="$(npu-smi info 2>&1)"; then
        printf '%s npu-smi failed: %s\n' "$(date -Is)" "$info" >>"$log_dir/runner.log"
        echo 0
        return
    fi
    printf '%s\n' "$info" | grep -cF 'No running processes found in NPU'
}

while true; do
    if [[ "$(idle_count)" -eq 8 ]]; then
        # Recheck after two seconds to avoid launching during device teardown.
        sleep 2
        if [[ "$(idle_count)" -eq 8 ]]; then
            break
        fi
    fi
    sleep "$poll_seconds"
done

printf 'running\n' >"$status_file"
printf '%s All NPUs idle; launching %s in %s\n' \
    "$(date -Is)" "$test_file" "$container" >>"$log_dir/runner.log"

docker exec -e "PYTHONPATH=$pythonpath" \
    -e "SGLANG_QSA_TEST_PROMPT_REPEAT=$prompt_repeat" "$container" /bin/bash -lc \
    "cd /sgl-workspace/sglang && python -m pytest '$test_file' -q" \
    >"$log_dir/pytest.log" 2>&1
rc=$?

printf '%s\n' "$rc" >"$log_dir/exit_code"
if [[ "$rc" -eq 0 ]]; then
    printf 'passed\n' >"$status_file"
else
    printf 'failed (%s)\n' "$rc" >"$status_file"
fi
printf '%s Test finished with exit code %s\n' "$(date -Is)" "$rc" \
    >>"$log_dir/runner.log"
exit "$rc"
