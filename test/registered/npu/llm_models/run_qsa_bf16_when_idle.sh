#!/usr/bin/env bash
# Run the BF16 QSA E2E test once the selected Ascend NPUs are idle.

set -u -o pipefail

container="${QSA_TEST_CONTAINER:-test-fby}"
test_file="${QSA_TEST_FILE:-/tmp/codex_qsa_bf16_20261013.py}"
pythonpath="${QSA_TEST_PYTHONPATH:-/sgl-workspace/sglang/python}"
prompt_repeat="${QSA_TEST_PROMPT_REPEAT:-320}"
log_dir="${QSA_TEST_LOG_DIR:-/tmp/codex_qsa_bf16_20261013_run}"
poll_seconds="${QSA_TEST_POLL_SECONDS:-5}"
idle_stable_seconds="${QSA_TEST_IDLE_STABLE_SECONDS:-15}"
visible_devices="${QSA_TEST_VISIBLE_DEVICES:-}"
test_url="${QSA_TEST_URL:-}"
indexer_dtype="${QSA_TEST_INDEXER_DTYPE:-bfloat16}"
launch_blocking="${QSA_TEST_ASCEND_LAUNCH_BLOCKING:-}"
idle_boards="${QSA_TEST_IDLE_BOARDS:-0 1 2 3 4 5 6 7}"
read -r -a board_ids <<<"$idle_boards"
if [[ "${#board_ids[@]}" -eq 0 ]]; then
    echo "QSA_TEST_IDLE_BOARDS must contain at least one NPU board ID" >&2
    exit 1
fi
mkdir -p "$log_dir"

exec 9>"$log_dir/lock"
if ! flock -n 9; then
    echo "A QSA test waiter is already running for $log_dir" >&2
    exit 1
fi

status_file="$log_dir/status"
printf 'waiting\n' >"$status_file"
printf '%s Waiting for NPU boards %s to become idle (visible chips: %s)\n' \
    "$(date -Is)" "$idle_boards" "${visible_devices:-default}" >>"$log_dir/runner.log"

idle_count() {
    local info
    if ! info="$(npu-smi info 2>&1)"; then
        printf '%s npu-smi failed: %s\n' "$(date -Is)" "$info" >>"$log_dir/runner.log"
        echo 0
        return
    fi
    local board count=0
    for board in "${board_ids[@]}"; do
        if printf '%s\n' "$info" | grep -qF "No running processes found in NPU $board "; then
            ((count += 1))
        fi
    done
    echo "$count"
}

while true; do
    if [[ "$(idle_count)" -eq "${#board_ids[@]}" ]]; then
        # A process can disappear from the process list before its HBM is
        # released. Keep the selected boards idle through a settling window.
        sleep "$idle_stable_seconds"
        if [[ "$(idle_count)" -eq "${#board_ids[@]}" ]]; then
            break
        fi
    fi
    sleep "$poll_seconds"
done

printf 'running\n' >"$status_file"
printf '%s Selected NPUs idle; launching %s in %s\n' \
    "$(date -Is)" "$test_file" "$container" >>"$log_dir/runner.log"

docker_env=(
    -e "PYTHONPATH=$pythonpath"
    -e "SGLANG_QSA_TEST_PROMPT_REPEAT=$prompt_repeat"
    -e "SGLANG_QSA_TEST_INDEXER_DTYPE=$indexer_dtype"
)
if [[ -n "$visible_devices" ]]; then
    docker_env+=(-e "ASCEND_RT_VISIBLE_DEVICES=$visible_devices")
fi
if [[ -n "$test_url" ]]; then
    docker_env+=(-e "SGLANG_QSA_TEST_URL=$test_url")
fi
if [[ -n "$launch_blocking" ]]; then
    docker_env+=(-e "ASCEND_LAUNCH_BLOCKING=$launch_blocking")
fi
docker exec "${docker_env[@]}" "$container" /bin/bash -lc \
    "cd /sgl-workspace/sglang && python -m pytest '$test_file' -s -q" \
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
