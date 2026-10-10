"""Compare decode performance with CP decode attention TP off and on."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)
from sglang.test.ascend.test_ascend_utils import run_bench_serving
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=7200, suite="nightly-8-npu-a3", nightly=True)

MODEL_PATH_ENV = "SGLANG_TEST_CP_DECODE_MODEL_PATH"
CP_DECODE_FLAG = "--enable-cp-decode-attn-tp"

# Keep the serving configuration aligned with test_npu_cp_decode_attn_tp.py.
TEST_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_BUFFSIZE": "1024",
    "DEEPEP_HCCL_BUFFSIZE": "1024",
    "SGLANG_ZBAL_LOCAL_MEM_SIZE": "0",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "0",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
}

COMMON_SERVER_ARGS = [
    "--tp-size",
    "8",
    "--attn-cp-size",
    "8",
    "--dp-size",
    "1",
    "--ep-size",
    "8",
    "--enable-prefill-cp",
    "--cp-strategy",
    "interleave",
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--quantization",
    "modelslim",
    "--kv-cache-dtype",
    "bfloat16",
    "--page-size",
    "128",
    "--mem-fraction-static",
    "0.8",
    "--chunked-prefill-size",
    "4096",
    "--prefill-max-requests",
    "8",
    "--max-running-requests",
    "8",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--watchdog-timeout",
    "600",
    "--random-seed",
    "0",
    "--disable-radix-cache",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--cuda-graph-backend-decode",
    "full",
    "--cuda-graph-bs-decode",
    "1",
    "2",
    "4",
    "8",
]


class TestNPUCpDecodeAttnTPPerformance(CustomTestCase):
    """Measure the decode gain from splitting attention linears across CP ranks.

    [Test Category] Context Parallel
    [Test Target] --enable-cp-decode-attn-tp
    """

    def test_decode_tpot_improves(self):
        model = os.environ.get(
            MODEL_PATH_ENV, DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH
        )

        results = {}
        with patch.dict(os.environ, TEST_ENVS):
            for enabled in (False, True):
                results[enabled] = run_bench_serving(
                    model=model,
                    dataset_name="generated-shared-prefix",
                    num_prompts=64,
                    gsp_num_groups=1,
                    gsp_prompts_per_group=64,
                    gsp_system_prompt_len=128,
                    gsp_question_len=128,
                    gsp_output_len=512,
                    request_rate=float("inf"),
                    max_concurrency=8,
                    seed=0,
                    other_server_args=COMMON_SERVER_ARGS
                    + ([CP_DECODE_FLAG] if enabled else []),
                    timeout_for_server_launch=1800,
                )

        disabled, enabled = results[False], results[True]
        for name, result in (("off", disabled), ("on", enabled)):
            print(
                f"CP decode attention TP {name}: "
                f"mean TPOT={result['mean_tpot_ms']:.2f} ms, "
                f"mean ITL={result['mean_itl_ms']:.2f} ms, "
                f"output throughput={result['output_throughput']:.2f} token/s, "
                f"mean TTFT={result['mean_ttft_ms']:.2f} ms",
                flush=True,
            )

        off_tpot = float(disabled["mean_tpot_ms"])
        on_tpot = float(enabled["mean_tpot_ms"])
        speedup = off_tpot / on_tpot
        throughput_gain = (
            float(enabled["output_throughput"]) / off_throughput - 1
        ) * 100
        print(
            f"CP decode attention TP: TPOT speedup={speedup:.3f}x, "
            f"output throughput change={throughput_gain:+.1f}%",
            flush=True,
        )
        self.assertGreater(
            speedup,
            1.0,
            f"CP decode attention TP did not improve mean TPOT: "
            f"off={off_tpot:.2f} ms, on={on_tpot:.2f} ms",
        )


if __name__ == "__main__":
    unittest.main()
