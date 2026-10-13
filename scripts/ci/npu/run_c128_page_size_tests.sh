#!/usr/bin/env bash

set -eo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer/bin/set_env.bash

cd "$repo_root"
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"

mode=${1:-}
if (($# > 0)); then
    shift
fi

case "$mode" in
    unit)
        python -m pytest -q -s \
            test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py::TestC128PageSize "$@"
        ;;
    e2e)
        : "${ASCEND_RT_VISIBLE_DEVICES:?Set the eight NPU IDs to use}"
        : "${SGLANG_DSV4_W8A8_MODEL_PATH:?Set the DeepSeek-V4-Flash W8A8 model path}"
        python -m pytest -q -s \
            test/registered/npu/basic_function/memory_and_scheduling/test_npu_c128_page_size.py "$@"
        ;;
    *)
        echo "Usage: $0 {unit|e2e}" >&2
        exit 2
        ;;
esac
