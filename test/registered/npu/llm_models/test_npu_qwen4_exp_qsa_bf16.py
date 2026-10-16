"""GSM8K accuracy regression for Qwen4-Exp (Qwen3.8-Flash-Next) QSA indexer
dtype on NPU.

The test serves one Qwen4-Exp model with the QSA query-sparse indexer stored
as bfloat16 (--qsa-indexer-dtype bfloat16). It checks the server configuration
and evaluates GSM8K to gate the bfloat16 indexer path instead of comparing
individual tokens or logprobs against a second server.
"""

import os
import unittest
from pathlib import Path

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=3600, suite="full-8-npu-a3")
register_npu_ci(est_time=7200, suite="nightly-8-npu-a3", nightly=True)

PARALLEL_SIZE = 8
MODEL_NAME = "Qwen3.8-Flash-Next"
MODEL_PATH_ENV = "SGLANG_TEST_QSA_BF16_MODEL_PATH"

TEST_ENVS = {
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_BUFFSIZE": "1024",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_USE_FIA": "1",
    "HCCL_EXEC_TIMEOUT": "200",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "0",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
}

SERVER_ARGS = [
    "--tp-size",
    str(PARALLEL_SIZE),
    "--trust-remote-code",
    "--device",
    "npu",
    "--dtype",
    "bfloat16",
    # The setting under test: store the compressed QSA indexer cache and index
    # query as bfloat16 (the main KV cache is unaffected).
    "--qsa-indexer-dtype",
    "bfloat16",
    # The A3 chunk_gated_delta_rule kernel only accepts bf16 initial_state;
    # fp32 state (the model config default) is rejected on Ascend910.
    "--mamba-ssm-dtype",
    "bfloat16",
    "--mem-fraction-static",
    "0.85",
    "--reasoning-parser",
    "qwen3-thinking",
    "--max-running-requests",
    "8",
    "--disable-radix-cache",
    "--random-seed",
    "0",
    # NEXTN speculative decoding (align with the passing qry-20260909f config).
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]


def resolve_model_path():
    override = os.environ.get(MODEL_PATH_ENV)
    candidates = (
        [Path(override)]
        if override
        else [
            Path("/mnt/paas/weights") / MODEL_NAME,
            Path("/home/weights") / MODEL_NAME,
            Path("/root/.cache/modelscope/hub/models/Qwen") / MODEL_NAME,
        ]
    )
    for path in candidates:
        if (path / "config.json").is_file():
            return str(path)
    raise FileNotFoundError(
        f"Set {MODEL_PATH_ENV} to a local Qwen4-Exp model directory; "
        f"searched: {candidates}. This test does not download model weights."
    )


class TestNPUQwen4ExpQSAIndexerBF16(GSM8KMixin, GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that --qsa-indexer-dtype bfloat16 keeps GSM8K accuracy
    on NPU for Qwen4-Exp (Qwen3.8-Flash-Next).

    [Test Category] Model
    [Test Target] --qsa-indexer-dtype bfloat16
    """

    model = resolve_model_path()
    timeout_for_server_launch = 1800  # Seconds, not milliseconds.
    other_args = SERVER_ARGS
    env = {**os.environ, **TEST_ENVS}

    # Qwen4-Exp ships only chat_template.jinja (no completion template), so the
    # raw-text completion few-shot from GSM8KAscendMixin makes the model emit EOS
    # immediately (score 0). Route GSM8K through sgl-eval (chat + boxed/sympy
    # grader), mirroring the B200 E2E test test_qwen4_exp_models.py in backend,
    # example count, and token budget. GSM8KMixin is listed first so its
    # test_gsm8k (sgl_eval) wins over GSM8KAscendMixin's completion version.
    gsm8k_backend = "sgl_eval"
    gsm8k_thinking = True
    gsm8k_num_examples = 5
    gsm8k_num_threads = 32
    gsm8k_max_tokens = 16384
    gsm8k_score_threshold = 0.90


if __name__ == "__main__":
    unittest.main()